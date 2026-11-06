
import math
from bisect import bisect_left, bisect_right
import numpy as np


# ─── Metric functions ────────────────────────────────────────────────────────
#
# Everything below scores a *single* monotonic one-to-one alignment between the
# known (ground-truth) boundary times and the predicted boundary times, computed
# with a Levenshtein / edit-distance DP.  That alignment is computed once per
# segment and then thresholded at each tolerance, so recall and precision always
# refer to the same set of matched pairs and no boundary is ever counted twice.


class _PrefixMax:
    """Fenwick tree over predicted-boundary indices holding (max score, argmax)."""

    __slots__ = ("n", "val", "arg")

    def __init__(self, n):
        self.n   = n
        self.val = [0.0] * (n + 1)   # 0.0 == "empty chain", all real scores are > 0
        self.arg = [-1]  * (n + 1)

    def update(self, i, value, arg):
        """Insert `value` at 1-based position i."""
        while i <= self.n:
            if value > self.val[i]:
                self.val[i] = value
                self.arg[i] = arg
            i += i & -i

    def query(self, i):
        """Max over 1-based positions [1..i]; returns (value, arg)."""
        best, best_arg = 0.0, -1
        while i > 0:
            if self.val[i] > best:
                best, best_arg = self.val[i], self.arg[i]
            i -= i & -i
        return best, best_arg
    
def levenshtein_align(known_boundaries, pred_boundaries, max_dist=100.0, gap_cost=None):
    """
    Levenshtein-style monotonic one-to-one alignment of two sorted boundary lists.

    Both lists are treated as symbol sequences in time order.  A predicted boundary
    may be *substituted* for a known boundary at cost |pred - known| (only allowed
    when that distance is <= max_dist), a known boundary may be *deleted* (a miss)
    and a predicted boundary may be *inserted* (a false alarm), each at cost
    gap_cost.  The alignment minimising the total edit cost is returned.

    Because the total cost is
        sum_matched |d| + gap_cost * (unmatched_known + unmatched_pred)
      = gap_cost * (n_known + n_pred) - sum_matched (2*gap_cost - |d|),
    minimising the cost is the same as picking the non-crossing set of pairs that
    maximises sum(2*gap_cost - |d|).  With gap_cost == max_dist every admissible
    pair carries a weight in [max_dist, 2*max_dist], so the alignment maximises the
    number of matched pairs first and minimises the total timing error second.
    That reduction is solved here in O(P log n_pred) over the P admissible pairs
    (a Fenwick prefix-max over predicted indices), instead of a full O(n*m) matrix.

    Args:
        known_boundaries: Sorted ground-truth boundary times (ms)
        pred_boundaries:  Sorted predicted boundary times (ms)
        max_dist:         Largest |pred - known| that may be matched at all (ms).
                          Must be >= the largest tolerance being scored.
        gap_cost:         Cost of a miss / false alarm (defaults to max_dist)

    Returns:
        List of (known_idx, pred_idx, signed_error) in increasing index order,
        where signed_error = pred - known (positive => prediction is late).
    """
    n, m = len(known_boundaries), len(pred_boundaries)
    if n == 0 or m == 0:
        return []

    gap = float(max_dist) if gap_cost is None else float(gap_cost)

    tree        = _PrefixMax(m)
    pair_score  = []
    pair_parent = []
    pair_known  = []
    pair_pred   = []
    best_score, best_pair = 0.0, -1

    for ki, k in enumerate(known_boundaries):
        lo = bisect_left(pred_boundaries,  k - max_dist)
        hi = bisect_right(pred_boundaries, k + max_dist)
        row_start = len(pair_score)

        for pj in range(lo, hi):
            weight = 2.0 * gap - abs(pred_boundaries[pj] - k)
            # best chain ending strictly before this pair (pred index < pj, known index < ki,
            # the latter guaranteed because this row is inserted only after it is finished)
            prev_score, prev_pair = tree.query(pj)
            score = prev_score + weight

            pair_score.append(score)
            pair_parent.append(prev_pair)
            pair_known.append(ki)
            pair_pred.append(pj)

            if score > best_score:
                best_score, best_pair = score, len(pair_score) - 1

        for idx in range(row_start, len(pair_score)):
            tree.update(pair_pred[idx] + 1, pair_score[idx], idx)

    matches = []
    idx = best_pair
    while idx >= 0:
        ki, pj = pair_known[idx], pair_pred[idx]
        matches.append((ki, pj, pred_boundaries[pj] - known_boundaries[ki]))
        idx = pair_parent[idx]
    matches.reverse()
    return matches


def nearest_distances(known_boundaries, pred_boundaries):
    """
    Nearest-neighbour distance from every known boundary to the closest predicted
    boundary (ms).  Purely descriptive — used for the error histogram, never for
    recall/precision (a nearest-neighbour match is many-to-one and would inflate
    both).  Returns a numpy array.
    """
    if not known_boundaries or not pred_boundaries:
        return np.empty(0, dtype=np.float64)

    known = np.asarray(known_boundaries, dtype=np.float64)
    pred  = np.asarray(pred_boundaries,  dtype=np.float64)
    pos   = np.searchsorted(pred, known)
    left  = np.abs(known - pred[np.clip(pos - 1, 0, len(pred) - 1)])
    right = np.abs(known - pred[np.clip(pos,     0, len(pred) - 1)])
    return np.minimum(left, right)


def best_shift(signed_errors, tolerance_ms, max_shift_ms):
    """
    Shift (ms) that would put the most matched pairs inside +-tolerance_ms.

    The mean/median signed error badly under-estimates a systematic lag, because the
    aligner re-matches around it and the surviving errors stay roughly symmetric.
    Scanning the error histogram for the densest tolerance-wide window measures the
    lag the metrics actually care about.  Returns 0.0 when there is nothing to scan.
    """
    if len(signed_errors) == 0 or max_shift_ms <= 0:
        return 0.0

    limit = int(max_shift_ms)
    edges = np.arange(-limit - 0.5, limit + 1.5, 1.0)
    hist, _ = np.histogram(signed_errors, bins=edges)
    window = np.ones(2 * int(tolerance_ms) + 1)
    counts = np.convolve(hist, window, mode="same")
    return float(np.arange(-limit, limit + 1)[int(np.argmax(counts))])


def r_value(hit_rate, over_segmentation):
    """
    Räsänen's R-value for segmentation quality (1.0 = perfect, penalises the
    over-segmentation that makes plain recall look good).

    Args:
        hit_rate: Recall as a fraction in [0, 1]
        over_segmentation: n_pred / n_known - 1
    """
    r1 = math.sqrt((1.0 - hit_rate) ** 2 + over_segmentation ** 2)
    r2 = (-over_segmentation + hit_rate - 1.0) / math.sqrt(2.0)
    return 1.0 - (abs(r1) + abs(r2)) / 2.0
