r'''
The peak loss: one objective, no blanks, no boundary labels.

## What it asks the model for

Every frame of the clip belongs to exactly one phoneme of the given sequence, and the posterior
of that phoneme is shaped like a hat over its segment:

        p(phone)  1.0  ______            ______            ______
                       \     /\     ....
                  0.5   \/      \/            \/                      <- boundaries
                       |  A   |  B   |  C   |

    peak of 1.0 across the interior of a phone,
    falling linearly to exactly 0.5 at each of its two boundaries,
    where the neighbouring phone is simultaneously rising through 0.5.

So the two phones that meet at a boundary CROSS there, and only there. Read along the top of
the picture and it is the V the boundary detector wants: the model's confidence
(top-1 minus top-2) dips to zero at a boundary and nowhere else. Read one phone's row and it
is the peak-in-the-middle shape a phone actually has.

## Why this and not CTC

CTC marks a phone with a spike and fills the space between spikes with blank. Where inside a
blank run the boundary lies is *not represented in the model's output at all*, so placement
inside that run is at chance - measured in this project at ~26 ms error against a 4.9-frame
mean blank run, i.e. exactly run/3. Every downstream trick (snapping onto spectral flux, a
learned boundary head, a partition CE) was an attempt to recover information CTC had already
thrown away, and the measurements say they recovered roughly none of it.

Here the boundary is not a by-product of the label sequence, it is the thing the output is
shaped around: it is the crossing point of two posteriors, which is a CONTINUOUS quantity that
`refine_onsets` reads off by linear interpolation. That is why a 5 ms frame grid is not the
resolution limit.

The soft 50/50 target also stops the saturation that CTC-with-a-frame-CE produced: the target
distribution has entropy, so the optimum is a calibrated posterior rather than a one-hot spike,
and `train_peak_kl` (loss minus target entropy) is the number that actually goes to zero.

## Where the segmentation comes from, given there are no boundary labels

From the model, by Viterbi, under `no_grad`, every step - classic hard-EM / flat start, the
same bootstrap HTK and MFA use. `viterbi_align` finds the monotonic assignment of frames to the
KNOWN phoneme sequence that maximises the model's own posteriors, subject to a minimum phone
duration. The targets are then built around that path and the model is trained towards them.

Cold start is handled by `prior_weight`: a diagonal band prior on the DP that pulls the path
towards a uniform-duration alignment while the model is still random, and is ramped to zero
over the first epochs. At weight 1 the alignment is essentially "divide the clip evenly by
phone count", which is precisely a flat start. Warm starting from a trained checkpoint? Set
`align_prior_epochs: 0`.

## Soft targets (`soft_target_mix`) - the entropy trap

Hard-EM trains the model towards ONE path, so wherever the aligner is unsure (gradual
transitions: glides, laterals, vowel onsets) the model is taught a committed guess and then
aligns more confidently on it next time. `alignment_occupancy` gives the posterior over ALL
paths on the same decoder instead; `soft_target_mix` blends it into the target (1 = pure
soft-EM). `soft_target_scale` < 1 tempers the path score. Lower values spread occupancy over
more legal alignments and eventually make the boundary target less precise.

**This knob is a trap at 1.0 and the measurements say so.** `_soft_target` multiplies the hat
by `1 - mix`, so at mix 1.0 the hat is not mixed down, it is arithmetically DELETED - and with
it `ramp_frames`, the 0.5 crossing, and the only thing keeping the target honest. The hat's
entropy is a function of the SEGMENTATION GEOMETRY alone, so it is pinned at ~0.17 nats and
cannot move. The occupancy's entropy is a function of the MODEL'S OWN CONFIDENCE, so it shrinks
as the model sharpens:

    sharper posteriors -> sharper occupancy -> harder target -> sharper posteriors

Measured over four consecutive warm-started runs at `soft_target_mix: 1.0`: target entropy
0.086 -> 0.070, `occupancy_peak` 0.962 -> 0.969, `frame_acc` 0.52 -> 0.69 (the model agreeing
with itself), while TIMIT `nobonus_f1_25ms` fell 72.7 -> 63.5 and `dip_f1_25ms` fell 39.8 ->
37.2 against a 29.9 random-init floor. Every self-referential number improved and the only one
scored against real labels collapsed. Leave this at 0 and use `soft_onset_mix`.

## Soft ONSETS (`soft_onset_mix`) - the same averaging, without the entropy

The fix is to change WHERE the occupancy is mixed in. Mixed into the frame class distribution
it destroys the hat; mixed into the ONSET POSITIONS it does not. `occupancy_to_onsets` reads
`E[onset_i]` and its spread off the same occupancy, the hat is rebuilt around those fractional
onsets, and the target's entropy goes back to being geometric and fixed.

Three things follow, and the first is the reason this is worth doing at all:

  - the target now carries SUB-FRAME boundary positions, which an integer hat cannot. That is
    what `refine_onsets` reads, so it is finally being trained against something it can see
    rather than always against a half-integer;
  - the boundary moves CONTINUOUSLY between steps instead of jumping a whole frame, which by
    itself damps the oscillation hard-EM produces;
  - `soft_onset_ramp_k` widens the ramp in proportion to the onset's own standard deviation, so
    a boundary the aligner is unsure about is taught softly instead of as a committed guess.
    That is a NEGATIVE feedback: a confused model gets softer targets, a confident one relaxes
    back to the `ramp_frames` floor. It is the variant the earlier soft-EM run never actually
    tested (at scale 0.3 its target came out SHARPER than the hat, so it tested "no ramp").

Honest limit: where the model is confidently WRONG - two thirds of the far misses have a
posterior onset spread under 10 ms - none of this helps. It stops new errors being locked in;
it does not repair old ones.

## Optional silence

Sequences from the phonemizer often do not spell out leading/trailing silence, and TIMIT clips
open with ~130 ms of it. Without a blank there is nothing to absorb that, so the first phone
would be dragged back to frame 0 - a guaranteed error on the first boundary of every clip.
`prepare_targets` therefore pads a SIL at each end and `viterbi_align` lets the path SKIP those
two states entirely (zero frames), the same way MFA's optional-silence states work.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e9
SIL_CLASS = 0



# ----------------------------------------------------------------------------- targets

def prepare_targets(targets, group_targets, target_lens, sil_class=SIL_CLASS,
                    optional_sil=True):
    '''
    Pad an optional SIL onto both ends of every target sequence.

    Returns:
        ext_targets [B, S+2], ext_groups [B, S+2], ext_lens [B],
        opt_head [B] bool, opt_tail [B] bool   (which of the two pads may be skipped)
    '''
    b, s = targets.shape
    device = targets.device

    # `SegmentsCollator` pads sequences with -100, the usual ignore_index. Nothing downstream
    # reads past `target_lens`, but the emission matrix is gathered for EVERY column, so a
    # negative id there is an out-of-bounds index into the class dimension - which on CUDA is
    # a device-side assert several kernels later, with a stack trace pointing somewhere else.
    # Clamp once, here, where the sequences enter the loss.
    targets = targets.clamp(min=0)
    group_targets = group_targets.clamp(min=0)

    if not optional_sil:
        return targets, group_targets, target_lens, \
            torch.zeros(b, dtype=torch.bool, device=device), \
            torch.zeros(b, dtype=torch.bool, device=device)

    idx = torch.arange(s, device=device)[None, :]
    last = (target_lens - 1).clamp(min=0)[:, None]
    # Only pad where the sequence does not already say SIL, otherwise a clip with a labelled
    # leading silence would get two SIL states and the DP would have to spend min_frames on
    # each of them.
    opt_head = targets.gather(1, torch.zeros_like(last)).squeeze(1) != sil_class
    opt_tail = targets.gather(1, last).squeeze(1) != sil_class

    ext = torch.full((b, s + 2), sil_class, dtype=targets.dtype, device=device)
    ext_g = torch.full((b, s + 2), sil_class, dtype=targets.dtype, device=device)
    shift = opt_head.long()[:, None]                       # 1 where a head pad was added
    pos = idx + shift
    ext.scatter_(1, pos, targets)
    ext_g.scatter_(1, pos, group_targets)
    # scatter wrote the padding slots too; re-stamp the tail pad after the real sequence.
    ext_lens = target_lens + opt_head.long() + opt_tail.long()
    tail_pos = (target_lens + opt_head.long())[:, None]
    ext.scatter_(1, tail_pos, torch.full_like(tail_pos, sil_class))
    ext_g.scatter_(1, tail_pos, torch.full_like(tail_pos, sil_class))
    return ext, ext_g, ext_lens, opt_head, opt_tail


# ----------------------------------------------------------------------------- the DP

@torch.no_grad()
def class_prior_normalise(log_probs, input_lens, smoothing=0.1):
    '''
    Posteriors -> scaled likelihoods for the aligner: `log p(c|x) - log p_bar(c)`.

    Raw posteriors can favor a globally frequent class. Dividing by the class prior makes a
    class predicted uniformly everywhere contribute little to the alignment score; only frames
    where a class is more likely than usual pull the boundary. `p_bar` is mixed with a uniform
    smoothing term so missing classes remain numerically usable.

    Alignment only: the CE still trains the raw posteriors against the resulting targets.
    '''
    b, t, c = log_probs.shape
    valid = (torch.arange(t, device=log_probs.device)[None, :] < input_lens[:, None])
    valid = valid.to(log_probs.dtype)[..., None]
    p_bar = (log_probs.exp() * valid).sum((0, 1)) / valid.sum().clamp(min=1.0)
    p_bar = (1.0 - smoothing) * p_bar + smoothing / c
    return log_probs - p_bar.log()[None, None, :]


@torch.no_grad()
def boundary_bonus(flux, input_lens, contrast_frames=17):
    '''
    Acoustic change -> per-frame onset bonus: `bonus[b, t]` is evidence that a phone STARTS
    at frame t. The torch/batched twin of BFA's `segmental.boundary_bonus`.

    Args:
        flux: [B, T] change ENTERING each frame (entry t compares frames t-1 and t, entry 0 is
            meaningless), e.g. `ComposedModel.encode_clip(..., return_flux=True)`.
        contrast_frames: moving-average window subtracted before scaling. Local contrast, not
            the raw flux, is what carries the boundary: a loud stretch of speech has high flux
            everywhere. 7 frames measured best in BFA at 13.125 ms frames; 17 is the same
            ~85 ms at 5 ms and measured best here (see the yaml).
    Returns:
        [B, T] float, zero at frame 0 and past each clip's end. Scaled by the clip's own flux
        std, so `boundary_bonus_weight` is in nats per standard deviation of change.
    '''
    b, t = flux.shape
    out = torch.zeros(b, t, device=flux.device)
    for i in range(b):
        n = int(input_lens[i])
        f = flux[i, 1:n].float()
        if f.numel() < 2:
            continue
        k = int(max(1, min(contrast_frames, f.numel())))
        pad = F.pad(f[None, None], (k // 2, k - 1 - k // 2), mode='replicate')
        avg = F.avg_pool1d(pad, k, stride=1)[0, 0]
        out[i, 1:n] = (f - avg) / (f.std() + 1e-6)
    return out


def _segmental_topology(log_probs, ext_targets, ext_lens, input_lens, min_frames, opt_head,
                        opt_tail, prior_weight, prior_sigma, prior_strength, class_prior_norm,
                        soft_frames, soft_cost, sil_skippable, sil_class, onset_bonus,
                        bonus_weight):
    '''
    Everything `viterbi_align` and `alignment_occupancy` share: the per-phone state chains,
    which phones may be skipped, the exit penalties, the emission and bonus terms. One copy,
    so the soft targets are always computed on exactly the decoder the hard ones are.
    '''
    b, t_padded, _ = log_probs.shape
    s = ext_targets.shape[1]
    d, k = int(min_frames), int(soft_frames)
    if d < 1 or k < 0:
        raise ValueError(f"need min_frames >= 1 and soft_frames >= 0, got {d}, {k}")
    n_sub = d + k
    device = log_probs.device
    NEGT = torch.tensor(NEG, device=device)

    # The recursions run once per FRAME, so they must not run over the batch's padding.
    t_max = max(1, min(t_padded, int(input_lens.max())))
    lp = log_probs[:, :t_max].float()
    if class_prior_norm:
        lp = class_prior_normalise(lp, input_lens)

    rows = torch.arange(b, device=device)
    pos = torch.arange(s, device=device)
    valid_phone = pos[None, :] < ext_lens[:, None]                          # [B, S]
    last = (ext_lens - 1).clamp(min=0)
    skippable = torch.zeros(b, s, dtype=torch.bool, device=device)
    if opt_head is not None:
        skippable[:, 0] |= opt_head
    if opt_tail is not None:
        skippable[rows, last] |= opt_tail
    if sil_skippable:
        skippable |= ext_targets == sil_class
    skippable &= valid_phone
    # Where a visited phone enters its chain: a short (skippable) SIL straight into its last,
    # self-looping state, so it needs one frame; everything else at state 0.
    enter_last = (skippable if sil_skippable else torch.zeros_like(skippable)) | (n_sub == 1)
    feasible = input_lens >= ((valid_phone & ~skippable).sum(1) * d).clamp(min=1)

    # Leaving phone i from sub-state j costs the soft states it skips; j < d-1 cannot leave.
    j_idx = torch.arange(n_sub, device=device)
    exit_pen = torch.where(j_idx >= d - 1, -(n_sub - 1 - j_idx).float() * float(soft_cost),
                           NEGT)                                             # [D]

    tgt = ext_targets.clamp(min=0)
    phone_ok = torch.where(valid_phone, 0.0, NEGT)                          # [B, S]
    prior_scale = 0.0
    if prior_weight > 0:
        prior_scale = prior_weight * prior_strength / (prior_sigma ** 2)
        idx_norm = (pos[None, :].float() + 0.5) / ext_lens[:, None].clamp(min=1).float()
        inv_len = 1.0 / input_lens.clamp(min=1).float()

    def emission(t):
        e = lp[:, t].gather(1, tgt) + phone_ok
        if prior_scale:
            e = e - prior_scale * (t * inv_len[:, None] - idx_norm).pow(2)
        return e[:, :, None]                                                 # [B, S, 1]

    if torch.is_tensor(bonus_weight):
        use_bonus = onset_bonus is not None and bool((bonus_weight != 0).any())
        bonus_w = bonus_weight.float()                                        # [B, S]
    else:
        use_bonus = onset_bonus is not None and bonus_weight != 0.0
        bonus_w = float(bonus_weight)
    bonus = onset_bonus[:, :t_max].float() if use_bonus else None
    return dict(t_max=t_max, n_sub=n_sub, rows=rows, last=last, skippable=skippable,
                enter_last=enter_last, feasible=feasible, exit_pen=exit_pen,
                emission=emission, use_bonus=use_bonus, bonus=bonus, bonus_w=bonus_w)


@torch.no_grad()
def viterbi_align(log_probs, ext_targets, ext_lens, input_lens, min_frames=2,
                  opt_head=None, opt_tail=None, prior_weight=0.0, prior_sigma=0.1,
                  prior_strength=1.0, class_prior_norm=False, soft_frames=0, soft_cost=1.0,
                  sil_skippable=False, sil_class=SIL_CLASS, onset_bonus=None,
                  bonus_weight=0.0):
    '''
    Monotonic forced alignment of `ext_targets` onto the frames.

    ## Topology (the segmental decoder's, `bournemouth_aligner/segmental.py`)

    Each phone is a chain of `min_frames + soft_frames` sub-states; only the last has a
    self-loop. The first `min_frames` must be visited. The `soft_frames` after them may be
    skipped at `soft_cost` nats each - a phone shorter than `min_frames + soft_frames` pays
    for every frame it is short by, so the duration prior is soft, which a constant per-phone
    cost can never be (every path enters the same number of phones and pays it equally).

    A SKIPPABLE phone may be jumped over at no cost. The optional SIL pads (`opt_head`,
    `opt_tail`) always are; with `sil_skippable` every SIL in the sequence is too - the
    phonemizer writes one at each comma whether or not the speaker paused - and a visited
    skippable phone needs only 1 frame. Without it, a punctuation SIL is forced to
    `min_frames`, i.e. a fake pause at every comma.

    `onset_bonus[b, t] * bonus_weight` is added to every transition INTO a phone at frame t
    (`bonus_weight` a float, or [B, S] for a per-phone weight),
    so posterior evidence and acoustic change are traded off in one global pass instead of
    snapping onsets greedily afterwards. See `boundary_bonus`.

    With soft_frames=0, sil_skippable=False and no bonus this is the plain min-duration chain.

    Args:
        log_probs: [B, T, C] log posteriors (NOT logits)
        ext_targets: [B, S] class ids, from `prepare_targets`
        ext_lens: [B] valid entries of ext_targets
        input_lens: [B] valid frames
        prior_weight: 0..1 on the diagonal band prior (cold start, see module docstring)
        class_prior_norm: align on log p(c|x) - log p_bar(c), the hybrid-HMM scaled
            likelihood, with p_bar the mean posterior over this batch's valid frames. See
            `class_prior_normalise` for the collapse it prevents.
    Returns:
        phone_idx [B, T] long: index into ext_targets for each frame (garbage past input_lens)
        feasible [B] bool: False where the clip cannot hold the mandatory frames, in which
            case that clip's row of phone_idx is meaningless and the caller must drop it.

    Cost: one python loop over T each way. Emissions are gathered per frame ([B, S]) rather
    than materialised per state, and backpointers are two bytes per (frame, phone), so memory
    does not grow with the number of sub-states.
    '''
    b, t_padded, _ = log_probs.shape
    s = ext_targets.shape[1]
    device = log_probs.device
    NEGT = torch.tensor(NEG, device=device)
    top = _segmental_topology(log_probs, ext_targets, ext_lens, input_lens, min_frames,
                              opt_head, opt_tail, prior_weight, prior_sigma, prior_strength,
                              class_prior_norm, soft_frames, soft_cost, sil_skippable,
                              sil_class, onset_bonus, bonus_weight)
    t_max, n_sub, rows, last = top['t_max'], top['n_sub'], top['rows'], top['last']
    skippable, enter_last, feasible = top['skippable'], top['enter_last'], top['feasible']
    exit_pen, emission = top['exit_pen'], top['emission']
    use_bonus, bonus, bonus_w = top['use_bonus'], top['bonus'], top['bonus_w']

    # --- t = 0: start in phone 0, or in phone 1 past a skippable phone 0
    alpha = torch.full((b, s, n_sub), NEG, device=device)
    entry_j = torch.where(enter_last, n_sub - 1, 0)                          # [B, S]
    alpha[rows, 0, entry_j[:, 0]] = 0.0
    if s > 1:
        start1 = torch.where(skippable[:, 0], 0.0, NEGT)
        alpha[rows, 1, entry_j[:, 1]] = torch.maximum(alpha[rows, 1, entry_j[:, 1]], start1)
    alpha = alpha + emission(0)

    # Backpointers. last_code, for the last sub-state: 0 stay, 1 from the sub-state before,
    # 2 entered. enter_src, for any entry: skip * n_sub + the exit sub-state of the source.
    last_code = torch.zeros(t_max, b, s, dtype=torch.uint8, device=device)
    enter_src = torch.zeros(t_max, b, s, dtype=torch.uint8, device=device)
    active = torch.arange(t_max, device=device)[None, :] < input_lens[:, None]    # [B, T]
    skip_mid = skippable[:, 1:-1] if s > 2 else None
    ent = torch.full((b, s), NEG, device=device)
    src = torch.zeros(b, s, dtype=torch.long, device=device)

    for t in range(1, t_max):
        exit_v, exit_j = (alpha + exit_pen).max(-1)                          # [B, S]
        ent.fill_(NEG)
        src.zero_()
        ent[:, 1:] = exit_v[:, :-1]
        src[:, 1:] = exit_j[:, :-1]
        if skip_mid is not None:
            cand = torch.where(skip_mid, exit_v[:, :-2], NEGT)
            take = cand > ent[:, 2:]
            ent[:, 2:] = torch.where(take, cand, ent[:, 2:])
            src[:, 2:] = torch.where(take, exit_j[:, :-2] + n_sub, src[:, 2:])
        if use_bonus:
            ent = ent + bonus[:, t:t + 1] * bonus_w

        stay = alpha[:, :, -1]
        best, code = stay, torch.zeros(b, s, dtype=torch.uint8, device=device)
        nxt = torch.empty_like(alpha)
        if n_sub > 1:
            nxt[:, :, 1:] = alpha[:, :, :-1]
            nxt[:, :, 0] = torch.where(enter_last, NEGT, ent)
            within = alpha[:, :, -2]
            m = within > best
            best = torch.where(m, within, best)
            code = torch.where(m, 1, code).to(torch.uint8)
        ent_last = torch.where(enter_last, ent, NEGT)
        m = ent_last > best
        best = torch.where(m, ent_last, best)
        code = torch.where(m, 2, code).to(torch.uint8)
        nxt[:, :, -1] = best
        nxt = nxt + emission(t)

        on = active[:, t]
        alpha = torch.where(on[:, None, None], nxt, alpha)
        last_code[t] = torch.where(on[:, None], code, 0).to(torch.uint8)
        enter_src[t] = src.to(torch.uint8)

    # --- end: leave the last phone, or the one before a skippable last phone
    exit_v, exit_j = (alpha + exit_pen).max(-1)
    p = last.clone()
    if s > 1:
        alt = (last - 1).clamp(min=0)
        use_alt = skippable[rows, last] & (last > 0) & (exit_v[rows, alt] > exit_v[rows, last])
        p = torch.where(use_alt, alt, p)
    j = exit_j[rows, p]

    phone_idx = torch.zeros(b, t_max, dtype=torch.long, device=device)
    for t in range(t_max - 1, -1, -1):
        phone_idx[:, t] = p
        if t == 0:
            break
        lc = last_code[t, rows, p].long()
        es = enter_src[t, rows, p].long()
        is_last = j == n_sub - 1
        entered = (is_last & (lc == 2)) | (~is_last & (j == 0))
        back_one = (is_last & (lc == 1)) | (~is_last & (j > 0))
        skip = es // n_sub
        p_new = torch.where(entered, p - 1 - skip, p)
        j_new = torch.where(entered, es % n_sub, torch.where(back_one, j - 1, j))
        on = t < input_lens              # past a clip's end its state is frozen
        p = torch.where(on, p_new, p)
        j = torch.where(on, j_new, j)

    if t_padded > t_max:                 # restore the caller's width; these frames are padding
        phone_idx = torch.cat(
            [phone_idx, phone_idx[:, -1:].expand(b, t_padded - t_max)], dim=1)
    return phone_idx, feasible


def alignment_occupancy(log_probs, ext_targets, ext_lens, input_lens, min_frames=2,
                        opt_head=None, opt_tail=None, prior_weight=0.0, prior_sigma=0.1,
                        prior_strength=1.0, class_prior_norm=False, soft_frames=0,
                        soft_cost=1.0, sil_skippable=False, sil_class=SIL_CLASS,
                        onset_bonus=None, bonus_weight=0.0, scale=1.0, chunk=128):
    '''
    Soft twin of `viterbi_align`: the posterior probability that frame t belongs to phone s,
    summed over EVERY alignment the decoder allows instead of read off the single best one.

    Same topology, emissions, duration penalties and onset bonus as `viterbi_align` (they come
    from the same `_segmental_topology`). The forward recursion runs in the log semiring
    (logsumexp where Viterbi takes max), giving log Z per clip; the occupancies are then
    d log Z / d emission[t, s] - each frame's emission enters a path exactly once, for the
    phone the path puts there, so that gradient IS the posterior occupancy. Autograd does the
    backward pass, which is the whole point: a hand-written beta recursion would be a second
    copy of the topology free to disagree with the first.

    `scale` multiplies the whole path score (emissions, penalties, bonus, prior) before the
    sum: 1 is the model's own posterior over alignments, and scale -> inf collapses onto the
    Viterbi path (the test for this function).

    Args: as `viterbi_align`, plus
        scale: path-score scale (inverse temperature)
        chunk: clips per recursion. The recursion is sequential in T, so fewer, larger chunks
            are faster; memory for the backward grows with it (measured: 108 clips x 2000
            frames x 60 phones, 6+8 sub-states: 1.4 GB, 1.6 s; Viterbi on the same: 1.1 s)
    Returns:
        gamma [B, T_padded, S] float, rows summing to 1 over s on valid frames of feasible
            clips, 0 elsewhere
        feasible [B] bool, as `viterbi_align`
    '''
    b, t_padded, _ = log_probs.shape
    s = ext_targets.shape[1]
    device = log_probs.device
    gamma = torch.zeros(b, t_padded, s, device=device)
    feasible_all = torch.zeros(b, dtype=torch.bool, device=device)
    scale = float(scale)
    for lo in range(0, b, chunk):
        sl = slice(lo, min(b, lo + chunk))
        sub = lambda x: None if x is None else x[sl]                       # noqa: E731
        bw = bonus_weight[sl] if torch.is_tensor(bonus_weight) else bonus_weight
        top = _segmental_topology(
            log_probs[sl].detach(), ext_targets[sl], ext_lens[sl], input_lens[sl], min_frames,
            sub(opt_head), sub(opt_tail), prior_weight, prior_sigma, prior_strength,
            class_prior_norm, soft_frames, soft_cost, sil_skippable, sil_class,
            sub(onset_bonus), bw)
        g, feas = _occupancy_chunk(top, ext_lens[sl], input_lens[sl], s, scale, device)
        gamma[sl, :g.shape[1]] = g
        feasible_all[sl] = feas
    return gamma, feasible_all


def _occupancy_chunk(top, ext_lens, input_lens, s, scale, device):
    t_max, n_sub, rows, last = top['t_max'], top['n_sub'], top['rows'], top['last']
    skippable, enter_last, feasible = top['skippable'], top['enter_last'], top['feasible']
    b = rows.shape[0]
    NEGT = torch.tensor(NEG, device=device)
    exit_pen = top['exit_pen'] * scale
    with torch.no_grad():
        em = torch.cat([top['emission'](t) for t in range(t_max)], dim=2)   # [B, S, T]
        em = (em * scale).permute(0, 2, 1).contiguous()                       # [B, T, S]
        bonus = top['bonus'] * scale if top['use_bonus'] else None
    bonus_w = top['bonus_w']
    active = torch.arange(t_max, device=device)[None, :] < input_lens[:, None]
    skip_mid = skippable[:, 1:-1] if s > 2 else None
    neg_col = torch.full((b, 1), NEG, device=device)

    with torch.enable_grad():
        em = em.requires_grad_(True)
        init = torch.full((b, s, n_sub), NEG, device=device)
        entry_j = torch.where(enter_last, n_sub - 1, 0)
        init[rows, 0, entry_j[:, 0]] = 0.0
        if s > 1:
            start1 = torch.where(skippable[:, 0], 0.0, NEGT)
            init[rows, 1, entry_j[:, 1]] = torch.maximum(init[rows, 1, entry_j[:, 1]], start1)
        alpha = init + em[:, 0, :, None]
        for t in range(1, t_max):
            exit_v = torch.logsumexp(alpha + exit_pen, dim=-1)                # [B, S]
            ent = torch.cat([neg_col, exit_v[:, :-1]], dim=1)
            if skip_mid is not None:
                cand = torch.where(skip_mid, exit_v[:, :-2], NEGT)
                ent = torch.cat([ent[:, :2], torch.logaddexp(ent[:, 2:], cand)], dim=1)
            if bonus is not None:
                ent = ent + bonus[:, t:t + 1] * bonus_w
            # The last sub-state: stay, step in from the one before it, or enter directly (a
            # short SIL). The others: enter at 0, or step along the chain. As viterbi_align.
            last_in = [alpha[:, :, -1], torch.where(enter_last, ent, NEGT)]
            if n_sub > 1:
                last_in.append(alpha[:, :, -2])
            last_v = torch.logsumexp(torch.stack(last_in, -1), -1)[:, :, None]
            if n_sub > 1:
                first = torch.where(enter_last, NEGT, ent)[:, :, None]
                nxt = torch.cat([first, alpha[:, :, :-2], last_v], dim=2)
            else:
                nxt = last_v
            nxt = nxt + em[:, t, :, None]
            alpha = torch.where(active[:, t, None, None], nxt, alpha)
        exit_v = torch.logsumexp(alpha + exit_pen, dim=-1)
        log_z = exit_v[rows, last]
        if s > 1:
            alt = exit_v[rows, (last - 1).clamp(min=0)]
            use_alt = skippable[rows, last] & (last > 0)
            log_z = torch.where(use_alt, torch.logaddexp(log_z, alt), log_z)
        ok = feasible & torch.isfinite(log_z) & (log_z > NEG / 2)
        gamma, = torch.autograd.grad(torch.where(ok, log_z, 0.0).sum(), em)
    frame_ok = active & ok[:, None]
    return gamma * frame_ok[:, :, None], ok


@torch.no_grad()
def path_to_onsets(phone_idx, input_lens, n_positions):
    '''
    Frame index at which each target position starts.

    Returns [B, n_positions + 1]; entry `i` is the first frame of phone `i` and the last entry
    is the clip length, so `onsets[i + 1] - onsets[i]` is phone `i`'s duration. A phone the
    path skipped (an unused optional SIL) gets a zero-length segment rather than a wrong one.
    '''
    b, t_max = phone_idx.shape
    device = phone_idx.device
    frame_ok = torch.arange(t_max, device=device)[None, :] < input_lens[:, None]
    # Frames past the clip are parked above every position so they are never counted.
    pi = torch.where(frame_ok, phone_idx, torch.full_like(phone_idx, n_positions + 1))
    cols = torch.arange(n_positions + 1, device=device)[None, None, :]
    return (pi[:, :, None] < cols).sum(dim=1)


@torch.no_grad()
def occupancy_to_onsets(gamma, chunk=16):
    '''
    Expected fractional onsets and their spread, from the forward-backward occupancy.

    The soft twin of `path_to_onsets`. That one COUNTS the frames the single best path put
    before each phone; this one sums the PROBABILITY of the same event. The topology is
    monotonic, so frame `t` belongs to a phone before `i` exactly when phone `i` starts after
    `t`, which makes the survival function a plain cumulative sum over phones:

        S[t, i]        = P(onset_i > t) = sum_{s < i} gamma[t, s]
        E[onset_i]     = sum_t S[t, i]
        E[onset_i^2]   = sum_t (2t + 1) S[t, i]          (X >= 0 and integer-valued)

    so both moments are reductions of a tensor that already exists. Three properties make the
    result a drop-in for `path_to_onsets` rather than an approximation of it: `E[onset_i]` is
    non-decreasing in `i` (the inner sum only gains non-negative terms), `E[onset_0] = 0`, and
    `E[onset_S] = sum_{t,s} gamma = T`, so the row still partitions the clip exactly.

    The moments are taken in float64, in batch chunks. This is not defensive: at T = 2000,
    `E[X^2]` is ~4e6 while the variance can be O(1), and float32 carries seven digits - the
    subtraction would lose the answer completely and come back NEGATIVE, so `sigma` would be
    NaN and every downstream ramp with it. `chunk` bounds the float64 temporary (16 clips x
    2000 frames x 67 positions = 17 MB).

    Args:
        gamma: [B, T, S] from `alignment_occupancy`, already zeroed past each clip's end and
            on infeasible clips, so no length mask is needed here.
    Returns:
        onsets [B, S + 1] float, sigma [B, S + 1] float, both in frames.
    '''
    b, t, s = gamma.shape
    device = gamma.device
    wgt = 2.0 * torch.arange(t, device=device, dtype=torch.float64) + 1.0
    mu = torch.zeros(b, s + 1, device=device, dtype=torch.float64)
    m2 = torch.zeros(b, s + 1, device=device, dtype=torch.float64)
    for lo in range(0, b, chunk):
        sl = slice(lo, min(b, lo + chunk))
        # Exclusive cumsum over phones: entry i is the mass on phones < i, i.e. S[t, i].
        surv = F.pad(gamma[sl].cumsum(-1), (1, 0)).double()             # [n, T, S + 1]
        mu[sl] = surv.sum(1)
        m2[sl] = (surv * wgt[None, :, None]).sum(1)
    sigma = (m2 - mu.pow(2)).clamp(min=0.0).sqrt()
    return mu.float(), sigma.float()


@torch.no_grad()
def onsets_to_phone_idx(onsets, ext_lens, t_max):
    '''
    Inverse of `path_to_onsets`: which phone each frame belongs to, given (fractional) onsets.

    A boundary sits at `onset - 0.5` in frame coordinates (see `peak_weights`), so frame `t`
    belongs to phone `i` exactly when `onset_i <= t + 0.5 < onset_{i+1}` - one `searchsorted`.
    `right=True` takes the LAST of a run of equal onsets, which is what makes a zero-length
    (skipped) phone unreachable instead of swallowing the frames of the phone after it.

    With integer onsets from a monotonic path this returns that path's `phone_idx` exactly, so
    it is a no-op on the hard-EM route; the self-test asserts the round trip.

    `onsets` must be non-decreasing along dim 1 - `searchsorted` does not check, it just
    returns nonsense - so callers blending two onset estimates must `cummax` first.
    '''
    t = torch.arange(t_max, device=onsets.device, dtype=onsets.dtype) + 0.5
    t = t.expand(onsets.shape[0], -1).contiguous()     # searchsorted copies it otherwise
    idx = torch.searchsorted(onsets.contiguous(), t, right=True) - 1
    return idx.clamp(min=0).minimum((ext_lens - 1).clamp(min=0)[:, None])


# ----------------------------------------------------------------------------- the shape

@torch.no_grad()
def peak_weights(phone_idx, onsets, ext_lens, ramp_frames=3.0):
    '''
    The hat shape, per frame.

    A boundary sits BETWEEN frames, at `onset - 0.5` in frame coordinates, so no frame ever
    carries the degenerate exactly-0.5 target and the crossing stays recoverable by
    interpolation. For a frame `t` inside phone `i`:

        dl = t - (onset_i     - 0.5)        distance to the phone's left boundary
        dr = (onset_{i+1} - 0.5) - t        distance to its right boundary
        w  = 0.5 + 0.5 * min(min(dl, dr) / ramp, 1)

    so `w` is 0.5 half a frame outside the phone, climbs over `ramp` frames, and sits at 1.0
    across the interior. The remaining `1 - w` goes to whichever NEIGHBOUR is closer. The ramp
    is clipped to half the segment, so a phone shorter than 2*ramp becomes a triangle and its
    peak is below 1.0 - which is the honest thing for a phone too short to be certain about.

    `ramp_frames` is the one number that trades identity against placement. Wide: mushy
    interiors, forgiving targets. Narrow: sharp, close to a hard label, and the interpolated
    crossing gets noisy. 3 frames = 15 ms each side at this frame rate, which is about the
    length of a real coarticulated transition. It may also be a [B, S + 1] tensor, one width
    per BOUNDARY, which is how `soft_onset_ramp_k` widens the ramp where the aligner is unsure.

    `onsets` may be integer (the Viterbi path) or float (the occupancy's expected onsets, see
    `occupancy_to_onsets`); the arithmetic here is float either way, and a fractional onset is
    exactly how a sub-frame boundary reaches the target.

    Returns:
        w [B, T] weight on the frame's own phone, partner [B, T] index of the other phone
    '''
    i = phone_idx
    j = (i + 1).clamp(max=onsets.shape[1] - 1)
    left = onsets.gather(1, i).float() - 0.5
    right = onsets.gather(1, j).float() - 0.5
    t = torch.arange(phone_idx.shape[1], device=phone_idx.device).float()[None, :]

    dl = t - left
    dr = right - t
    # One ramp width per BOUNDARY, not per phone, so the two phones meeting at a boundary use
    # the same width there and the crossing stays at exactly 0.5. A per-phone width would let
    # them disagree and move the crossing, which is the one quantity `refine_onsets` reads.
    half = ((right - left) / 2).clamp(min=1e-3)
    if torch.is_tensor(ramp_frames):
        rl = torch.minimum(ramp_frames.gather(1, i).float(), half).clamp(min=1e-3)
        rr = torch.minimum(ramp_frames.gather(1, j).float(), half).clamp(min=1e-3)
    else:
        rl = rr = torch.minimum(torch.full_like(dl, float(ramp_frames)), half).clamp(min=1e-3)
    w = 0.5 + 0.5 * torch.minimum(dl / rl, dr / rr).clamp(0.0, 1.0)

    # The neighbour is the nearest phone the path actually VISITED. A skipped phone (an unused
    # optional or punctuation SIL) has no frames, so the phone across the boundary is the one
    # beyond it - and a skipped pad at a clip edge means there is no neighbour at all, where
    # `i - 1` used to hand ~40% of frame 0's target to a SIL that was never there.
    s = onsets.shape[1] - 1
    pos = torch.arange(s, device=phone_idx.device)[None, :]
    # Visited = the phone actually CAPTURES a frame, not merely that its duration is positive.
    # Its first frame is `ceil(onset - 0.5)`, so it is non-empty iff its first frame differs
    # from its successor's. With integer onsets `ceil(n - 0.5) == n`, so this is bit-identical
    # to the old `diff > 0`; with FRACTIONAL onsets that test would mark a skipped optional SIL
    # of expected duration 0.02 as visited, and frame 0 would go back to handing ~40% of its
    # target to a SIL that is not there.
    first = torch.ceil(onsets.float() - 0.5)
    visited = (first[:, 1:] > first[:, :-1]) & (pos < ext_lens[:, None])
    before = F.pad(torch.where(visited, pos, -1), (1, 0), value=-1)[:, :-1]
    prev_visited = torch.cummax(before, dim=1).values                        # [B, S]
    after = F.pad(torch.where(visited, pos, s), (0, 1), value=s)[:, 1:]
    next_visited = torch.flip(torch.cummin(torch.flip(after, [1]), dim=1).values, [1])
    partner = torch.where(dr <= dl, next_visited.gather(1, i), prev_visited.gather(1, i))
    has_partner = (partner >= 0) & (partner < ext_lens[:, None])
    w = torch.where(has_partner, w, torch.ones_like(w))
    partner = partner.clamp(min=0)
    partner = torch.minimum(partner, (ext_lens - 1)[:, None])
    return w, partner


class PeakLoss(nn.Module):
    '''
    The whole training objective. One term, applied to the phoneme head and (at a fixed
    weight) to the coarse phoneme-group head, which is the same target read at a different
    granularity rather than a second objective with its own schedule.

    Args:
        min_frames: shortest phone the aligner may produce, in frames (2 = 10 ms)
        ramp_frames: see `peak_weights`
        group_weight: weight on the group head's copy of the loss. 0 leaves the head untrained
            (and then DDP needs find_unused_parameters)
        optional_sil: allow a skippable SIL at each end of the sequence
        soft_frames / soft_cost / sil_skippable / bonus_weight / contrast_frames: the
            segmental decoder's topology and flux bonus, see `viterbi_align` and
            `boundary_bonus`. All off by default.
        soft_target_mix: mix the occupancy into the CLASS DISTRIBUTION. Leave at 0; at 1 it
            deletes the hat and the target's entropy becomes a free variable. See the module
            docstring for the four runs that measured what that does.
        soft_onset_mix: mix the occupancy into the ONSET POSITIONS instead. 0 = the Viterbi
            path's integer onsets, i.e. bit-identical to hard-EM.
        soft_onset_ramp_k: widen each boundary's ramp by `k * sigma`, sigma being that onset's
            own standard deviation under the occupancy. 0 = a constant `ramp_frames`.
        soft_onset_ramp_max: cap on the widened ramp, in frames. A ramp wider than half a
            phone turns it into a triangle whose peak is below 1.0, and the aligner then
            scores short phones as CHEAPER (see `peak_weights`), so this cap is what stops
            ramp widening from feeding the duration squeeze. None = no widening beyond
            `ramp_frames`.
    '''

    def __init__(self, min_frames=2, ramp_frames=3.0, group_weight=0.3, optional_sil=True,
                 prior_sigma=0.1, prior_strength=1.0, sil_class=SIL_CLASS,
                 class_prior_norm=False, soft_frames=0, soft_cost=1.0, sil_skippable=False,
                 bonus_weight=0.0, contrast_frames=17, soft_target_mix=0.0,
                 soft_target_scale=0.3, soft_onset_mix=0.0, soft_onset_ramp_k=0.0,
                 soft_onset_ramp_max=None):
        super().__init__()
        # 0 = the hat target alone (hard-EM). >0 mixes in the forward-backward occupancy at
        # path-score scale `soft_target_scale`, see `alignment_occupancy` and `forward`.
        self.soft_target_mix = float(soft_target_mix)
        self.soft_target_scale = float(soft_target_scale)
        # The same occupancy, used on the ONSETS instead of on the class distribution, so the
        # hat survives and the target keeps its fixed geometric entropy. See the module
        # docstring; these two are mutually exclusive and `forward` enforces that.
        self.soft_onset_mix = float(soft_onset_mix)
        self.soft_onset_ramp_k = float(soft_onset_ramp_k)
        self.soft_onset_ramp_max = None if soft_onset_ramp_max is None \
            else float(soft_onset_ramp_max)
        self.class_prior_norm = bool(class_prior_norm)
        self.soft_frames = int(soft_frames)
        self.soft_cost = float(soft_cost)
        self.sil_skippable = bool(sil_skippable)
        self.bonus_weight = float(bonus_weight)
        self.contrast_frames = int(contrast_frames)
        self.min_frames = int(min_frames)
        self.ramp_frames = float(ramp_frames)
        self.group_weight = float(group_weight)
        self.optional_sil = bool(optional_sil)
        self.prior_sigma = float(prior_sigma)
        self.prior_strength = float(prior_strength)
        self.sil_class = int(sil_class)

    def align(self, log_probs, targets, group_targets, target_lens, input_lens,
              prior_weight=0.0, flux=None, with_occupancy=False):
        '''
        The no-grad half: everything needed to build a target, and nothing else.

        `flux` [B, T] is the acoustic change entering each frame. It only matters when
        `bonus_weight` is non-zero, and is then required: aligning without it would silently
        be a different decoder from the one configured.

        `with_occupancy` adds 'occupancy' [B, T, S]: the forward-backward posterior of each
        phone position per frame, on the same decoder, at `soft_target_scale`; plus
        'soft_onsets' and 'onset_sigma' [B, S + 1], its expected onsets and their spread.

        'onsets' STAYS the integer Viterbi tensor whatever else is computed: `align_batch`,
        `refine_onsets` and `_boundary_frames` all index with it, so its dtype is API.
        '''
        ext, ext_g, ext_lens, opt_head, opt_tail = prepare_targets(
            targets, group_targets, target_lens, self.sil_class, self.optional_sil)
        bonus = None
        if self.bonus_weight:
            if flux is None:
                raise ValueError("bonus_weight is set but no flux was passed to align()")
            bonus = boundary_bonus(flux, input_lens, self.contrast_frames)
        phone_idx, feasible = viterbi_align(
            log_probs, ext, ext_lens, input_lens, min_frames=self.min_frames,
            opt_head=opt_head, opt_tail=opt_tail, prior_weight=prior_weight,
            prior_sigma=self.prior_sigma, prior_strength=self.prior_strength,
            class_prior_norm=self.class_prior_norm, soft_frames=self.soft_frames,
            soft_cost=self.soft_cost, sil_skippable=self.sil_skippable,
            sil_class=self.sil_class, onset_bonus=bonus, bonus_weight=self.bonus_weight)
        onsets = path_to_onsets(phone_idx, input_lens, ext.shape[1])
        out = {'ext': ext, 'ext_groups': ext_g, 'ext_lens': ext_lens,
               'phone_idx': phone_idx, 'onsets': onsets, 'feasible': feasible,
               'opt_head': opt_head, 'opt_tail': opt_tail}
        if with_occupancy:
            out['occupancy'], occ_ok = alignment_occupancy(
                log_probs, ext, ext_lens, input_lens, min_frames=self.min_frames,
                opt_head=opt_head, opt_tail=opt_tail, prior_weight=prior_weight,
                prior_sigma=self.prior_sigma, prior_strength=self.prior_strength,
                class_prior_norm=self.class_prior_norm, soft_frames=self.soft_frames,
                soft_cost=self.soft_cost, sil_skippable=self.sil_skippable,
                sil_class=self.sil_class, onset_bonus=bonus, bonus_weight=self.bonus_weight,
                scale=self.soft_target_scale)
            out['soft_onsets'], out['onset_sigma'] = occupancy_to_onsets(out['occupancy'])
            out['feasible'] = feasible & occ_ok
        return out

    def forward(self, log_probs, group_log_probs, targets, group_targets,
                target_lens, input_lens, prior_weight=0.0, alignment=None, flux=None):
        '''
        Args:
            log_probs: [B, T, C] log posteriors over phoneme classes
            group_log_probs: [B, T, G] log posteriors over phoneme groups (may be None)
            targets / group_targets: [B, S] label ids
            target_lens / input_lens: [B]
            prior_weight: diagonal prior weight for this step (ramped by the trainer)
            alignment: reuse a dict from `align()` instead of recomputing it
            flux: [B, T] acoustic change for the onset bonus, see `align`
        Returns:
            (loss, stats dict, alignment dict)
        '''
        soft_q = self.soft_target_mix > 0
        soft_on = self.soft_onset_mix > 0
        if soft_q and soft_on:
            raise ValueError(
                "soft_target_mix and soft_onset_mix are two different uses of the same "
                "occupancy - mixing it into the class distribution (which deletes the hat and "
                "its fixed entropy) or into the onsets (which does not). Pick one; see the "
                "module docstring.")
        need_occ = soft_q or soft_on
        with torch.no_grad():
            if alignment is None or (need_occ and 'occupancy' not in alignment):
                alignment = self.align(log_probs.detach(), targets, group_targets,
                                       target_lens, input_lens, prior_weight, flux=flux,
                                       with_occupancy=need_occ)
            ext, ext_g = alignment['ext'], alignment['ext_groups']
            ext_lens, phone_idx = alignment['ext_lens'], alignment['phone_idx']
            onsets, feasible = alignment['onsets'], alignment['feasible']

            ramp, onsets_hat = self.ramp_frames, onsets
            if soft_on:
                m = self.soft_onset_mix
                onsets_hat = (1 - m) * onsets.float() + m * alignment['soft_onsets']
                # `searchsorted` does not check sortedness, it just returns nonsense, and a
                # float blend can produce a -1e-7 step. Both guards are cheap.
                onsets_hat = torch.cummax(onsets_hat.clamp(min=0.0), dim=1).values
                onsets_hat = torch.minimum(onsets_hat, input_lens[:, None].float())
                # Re-derive the frame->phone assignment from the onsets actually used. Keeping
                # the Viterbi one would let a soft onset move past a frame that path assigned
                # elsewhere: `min(dl, dr)` goes negative, the clamp pins w to exactly 0.5 over
                # a stretch, and the crossing - the whole point - is flat instead of crossing.
                phone_idx = onsets_to_phone_idx(onsets_hat, ext_lens, log_probs.shape[1])
                if self.soft_onset_ramp_k > 0:
                    r_max = self.ramp_frames if self.soft_onset_ramp_max is None \
                        else self.soft_onset_ramp_max
                    ramp = (self.ramp_frames
                            + self.soft_onset_ramp_k * alignment['onset_sigma']).clamp(
                                self.ramp_frames, float(r_max))

            w, partner = peak_weights(phone_idx, onsets_hat, ext_lens, ramp)
            frame_ok = (torch.arange(log_probs.shape[1], device=log_probs.device)[None, :]
                        < input_lens[:, None]) & feasible[:, None]
            mask = frame_ok.to(log_probs.dtype)
            n_frames = mask.sum().clamp(min=1.0)

            main_cls = ext.gather(1, phone_idx)
            part_cls = ext.gather(1, partner)
            main_grp = ext_g.gather(1, phone_idx)
            part_grp = ext_g.gather(1, partner)
            w = w.to(log_probs.dtype)

            def _soft_target(n_cls, main, part, labels):
                '''(1 - mix) * the hat + mix * the occupancy, as a [B, T, n_cls] distribution.'''
                q = torch.zeros(*main.shape, n_cls, device=main.device)
                q.scatter_add_(2, main[:, :, None], (1 - self.soft_target_mix) * w.float()[:, :, None])
                q.scatter_add_(2, part[:, :, None],
                               (1 - self.soft_target_mix) * (1 - w.float())[:, :, None])
                occ = alignment['occupancy'][:, :main.shape[1]]
                idx = labels[:, None, :].expand(-1, main.shape[1], -1)
                return q.scatter_add_(2, idx, self.soft_target_mix * occ)

            q_main = q_grp = None
            if soft_q:
                q_main = _soft_target(log_probs.shape[2], main_cls, part_cls, ext)
                if group_log_probs is not None:
                    q_grp = _soft_target(group_log_probs.shape[2], main_grp, part_grp, ext_g)

        def _ce(lp, main, part, q=None):
            if q is not None:
                return -((q * lp.float()).sum(-1) * mask).sum() / n_frames
            lp_main = lp.gather(2, main[:, :, None]).squeeze(2)
            lp_part = lp.gather(2, part[:, :, None]).squeeze(2)
            return -((w * lp_main + (1 - w) * lp_part) * mask).sum() / n_frames

        loss_main = _ce(log_probs, main_cls, part_cls, q_main)
        loss = loss_main
        loss_groups = None
        if group_log_probs is not None and self.group_weight > 0:
            loss_groups = _ce(group_log_probs, main_grp, part_grp, q_grp)
            loss = loss + self.group_weight * loss_groups

        with torch.no_grad():
            # Entropy of the target itself. The CE cannot go below it, so `peak_kl` is the
            # part that can actually reach zero - read that, not the raw loss.
            wc = w.float().clamp(1e-6, 1 - 1e-6)
            hat_ent = -(wc * wc.log() + (1 - wc) * (1 - wc).log())
            if soft_q:
                qc = q_main.clamp(min=1e-12)
                ent = -(q_main * qc.log()).sum(-1)
            else:
                ent = hat_ent
            ent = (ent * mask).sum() / n_frames
            # The hat's entropy depends on the SEGMENTATION GEOMETRY only, never on the model,
            # so it is the fixed reference the real target entropy is read against. The ratio
            # is the single number that says whether the target is running away with the
            # model's confidence: 1.0 = pure hat, ~0.4 = what `soft_target_mix: 1.0` produced.
            hat_ent = (hat_ent * mask).sum() / n_frames
            durations = (onsets[:, 1:] - onsets[:, :-1]).float()
            seg_ok = (torch.arange(durations.shape[1], device=durations.device)[None, :]
                      < ext_lens[:, None]) & feasible[:, None]
            n_seg = seg_ok.sum().clamp(min=1)
            # The collapse alarm counts real phones only: a SIL may legitimately be 1 frame.
            speech = seg_ok & (ext != self.sil_class)
            stats = {
                'peak_ce': loss_main.detach(),
                'peak_kl': loss_main.detach() - ent,
                'target_entropy': ent,
                'hat_entropy': hat_ent,
                'target_entropy_ratio': ent / hat_ent.clamp(min=1e-6),
                # How much of the gradient is about BOUNDARIES at all rather than phone
                # identity. Falls as phones get longer, so it also reads the corpus: ~0.4 at
                # libri's 16-frame mean phone, ~0.2 on an isolated-word source.
                'ramp_frac': (((w.float() < 1.0 - 1e-6).float()) * mask).sum() / n_frames,
                'mean_phone_frames': (durations * seg_ok).sum() / n_seg,
                'min_dur_frac': (((durations > 0) & (durations <= self.min_frames)
                                  & speech).sum() / speech.sum().clamp(min=1)).float(),
                'skipped_sil_frac': ((durations == 0) & seg_ok).sum().float() / n_seg,
                'infeasible_frac': 1.0 - feasible.float().mean(),
                # Identity guard, replacing PER: how often the argmax names the phone the
                # aligner put there. No blank to collapse, so greedy PER measured argmax
                # flicker instead of identity; this does not.
                'frame_acc': ((log_probs.detach().argmax(-1) == main_cls).float() * mask).sum()
                / n_frames,
            }
            if loss_groups is not None:
                stats['peak_ce_groups'] = loss_groups.detach()
            if need_occ:
                # How soft the occupancy is: 1.0 = no different from Viterbi.
                occ = alignment['occupancy'][:, :mask.shape[1]]
                stats['occupancy_peak'] = (occ.max(-1).values * mask).sum() / n_frames
            if soft_on:
                # Interior boundaries only: position 0 is the clip start and position
                # `ext_lens` is its end, neither of which is a boundary anybody places.
                bpos = torch.arange(onsets.shape[1], device=onsets.device)[None, :]
                b_ok = (bpos >= 1) & (bpos < ext_lens[:, None]) & feasible[:, None]
                n_b = b_ok.sum().clamp(min=1)
                # Reads 0.0 if the occupancy at this `soft_target_scale` is just Viterbi, in
                # which case soft onsets are a no-op and the knob is doing nothing.
                shift = (alignment['soft_onsets'] - onsets.float()).abs()
                stats['onset_shift_frames'] = (shift * b_ok).sum() / n_b
                stats['onset_sigma_frames'] = (alignment['onset_sigma'] * b_ok).sum() / n_b
        return loss, stats, alignment


# ----------------------------------------------------------------------------- readout

@torch.no_grad()
def refine_onsets(log_probs, seq, onsets):
    '''
    Sub-frame boundary times from the crossing of two posteriors.

    The training target makes phone `i` and phone `i+1` cross exactly at their shared boundary,
    so the boundary is where `logp(i+1) - logp(i)` changes sign, and linear interpolation of
    that difference between the two frames either side of the crossing puts it at a fraction of
    a frame. At 5 ms frames this is the difference between a 2.5 ms quantisation floor and
    none - it is the reason the loss was shaped this way rather than with hard labels.

    Args:
        log_probs: [T, C] for one clip
        seq: [S] the (extended) target sequence
        onsets: [S + 1] integer onsets from `path_to_onsets`
    Returns:
        [S] float frame positions. Position 0 stays at its integer onset (nothing to cross).
    '''
    out = onsets[:-1].float().clone()
    t_max = log_probs.shape[0]
    prev = -1                                          # last phone the path visited
    for i in range(int(seq.shape[0])):
        if onsets[i + 1] == onsets[i]:
            continue                                   # skipped: no frames, no boundary
        left_i, prev = prev, i
        b = int(onsets[i])
        if left_i < 0 or b <= 0 or b >= t_max:
            continue                                   # first visited phone, or a clip edge
        # The phone across the boundary is the previous VISITED one, not seq[i - 1]: that may
        # be a skipped SIL, whose posterior has nothing to do with this crossing.
        left, right = int(seq[left_i]), int(seq[i])
        if left == right:
            continue
        g0 = (log_probs[b - 1, right] - log_probs[b - 1, left]).item()
        g1 = (log_probs[b, right] - log_probs[b, left]).item()
        if g0 < 0 < g1:                                 # the expected bracket
            out[i] = (b - 1) + g0 / (g0 - g1)
        else:
            out[i] = b - 0.5                            # no crossing here: mid-way is the
        # honest answer, and it matches the integer reading to within half a frame.
    return out


@torch.no_grad()
def confidence_dip(log_probs):
    '''
    The V, read WITHOUT the phoneme sequence: `1 - (top1 - top2)` per frame.

    This is the unsupervised boundary signal the shape was designed to create, and the probe
    scores it separately for exactly that reason. If `dip_f1` is near the forced-aligned
    number then the model can be used to segment audio it has no transcript for; if it is far
    below, the hat shape has not formed and the forced score is coming from the DP rather than
    from the posteriors.
    '''
    top2 = log_probs.exp().topk(2, dim=-1).values
    return 1.0 - (top2[..., 0] - top2[..., 1])


@torch.no_grad()
def embedding_delta(embeddings):
    '''
    Cosine change between adjacent frames, indexed so that entry `t` is the change ENTERING
    frame `t` (i.e. a peak at `t` means a boundary between `t-1` and `t`). Entry 0 is zero.

    This is spectral flux on whatever tensor it is given. Short-receptive-field features retain
    local changes more clearly than features that have already mixed time.
    '''
    e = F.normalize(embeddings.float(), dim=-1)
    delta = 1.0 - (e[1:] * e[:-1]).sum(-1)
    return F.pad(delta, (1, 0), value=0.0)


@torch.no_grad()
def parabolic_refine(signal, indices):
    '''
    Sub-frame position of a discrete peak, by fitting a parabola through its two neighbours.

    Needed because the trained shape is SYMMETRIC about a boundary that sits between two
    frames: the two frames either side of it carry equal confidence, so the integer peak is
    half a frame off by construction, in one direction or the other. The parabola through
    (t-1, t, t+1) puts it back in the middle.
    '''
    x = signal.detach().float()
    out = []
    n = x.numel()
    for i in indices:
        if 0 < i < n - 1:
            a, b, c = float(x[i - 1]), float(x[i]), float(x[i + 1])
            denom = a - 2 * b + c
            shift = 0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0.0
            out.append(i + max(-0.5, min(0.5, shift)))
        else:
            out.append(float(i))
    return out


@torch.no_grad()
def peak_pick(signal, k, min_distance=2):
    '''Top-`k` local maxima of a 1-D signal, at least `min_distance` frames apart.'''
    x = signal.detach().float().cpu()
    n = x.numel()
    if n < 3 or k <= 0:
        return []
    is_peak = torch.zeros(n, dtype=torch.bool)
    is_peak[1:-1] = (x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:])
    cand = torch.nonzero(is_peak).flatten().tolist()
    cand.sort(key=lambda i: -float(x[i]))
    chosen = []
    for i in cand:
        if all(abs(i - j) >= min_distance for j in chosen):
            chosen.append(i)
            if len(chosen) >= k:
                break
    return sorted(chosen)


if __name__ == "__main__":
    torch.manual_seed(0)
    print("--- peak_weights shape ---")
    # One clip, 20 frames, 3 phones with onsets 0, 6, 14, end 20.
    onsets = torch.tensor([[0, 6, 14, 20, 20]])
    phone_idx = torch.tensor([[0] * 6 + [1] * 8 + [2] * 6])
    w, partner = peak_weights(phone_idx, onsets, torch.tensor([3]), ramp_frames=3.0)
    print("w:", [round(float(v), 3) for v in w[0]])
    print("partner:", partner[0].tolist())
    # At the boundary (frames 5 and 6) the two phones must be symmetric about 0.5 and the
    # crossing must therefore land at 5.5, which is onset 6 - 0.5.
    assert abs(float(w[0, 5]) - float(w[0, 6])) < 1e-6, (w[0, 5], w[0, 6])
    assert 0.5 < float(w[0, 5]) < 0.7
    assert float(w[0, 2]) == 1.0 and float(w[0, 10]) == 1.0
    assert partner[0, 5].item() == 1 and partner[0, 6].item() == 0
    assert float(w[0, 0]) == 1.0, "no left neighbour -> no fade at the clip edge"

    print("\n--- viterbi recovers a planted alignment ---")
    T, C, S = 60, 10, 4
    true_onsets = [0, 17, 31, 44, 60]
    logits = torch.full((1, T, C), -3.0)
    seq = torch.tensor([[2, 5, 7, 3]])
    grp = torch.tensor([[1, 1, 2, 2]])
    for i in range(S):
        logits[0, true_onsets[i]:true_onsets[i + 1], seq[0, i]] = 3.0
    lp = F.log_softmax(logits, dim=-1)
    ext, ext_g, ext_lens, oh, ot = prepare_targets(seq, grp, torch.tensor([S]))
    print("extended:", ext[0].tolist(), "len", int(ext_lens[0]), "opt head/tail:",
          bool(oh[0]), bool(ot[0]))
    pi, feasible = viterbi_align(lp, ext, ext_lens, torch.tensor([T]), min_frames=2,
                                 opt_head=oh, opt_tail=ot)
    got = path_to_onsets(pi, torch.tensor([T]), ext.shape[1])[0].tolist()
    print("onsets:", got, " (optional SILs at both ends should be empty)")
    assert bool(feasible[0])
    assert got[0] == got[1] == 0, ("leading SIL should be skipped", got)
    assert got[-1] == got[-2] == T, ("trailing SIL should be skipped", got)
    assert got[1:6] == true_onsets, (got, true_onsets)

    print("\n--- sub-frame refinement ---")
    refined = refine_onsets(lp[0], ext[0, :int(ext_lens[0])],
                            torch.tensor(got[:int(ext_lens[0]) + 1]))
    print("refined:", [round(float(v), 2) for v in refined])

    print("\n--- soft onsets: the occupancy's expectation, and the round trip ---")
    hard_on = path_to_onsets(pi, torch.tensor([T]), ext.shape[1])
    # `onsets_to_phone_idx` must invert `path_to_onsets` exactly, or swapping it in for the
    # hard-EM route would silently move every frame's target.
    back = onsets_to_phone_idx(hard_on.float(), ext_lens, T)
    assert torch.equal(back, pi[:, :T]), (back[0].tolist(), pi[0, :T].tolist())
    print("round trip path_to_onsets -> onsets_to_phone_idx: identical")
    for sc in (0.3, 1.0, 50.0):
        g, ok = alignment_occupancy(lp, ext, ext_lens, torch.tensor([T]), min_frames=2,
                                    opt_head=oh, opt_tail=ot, scale=sc)
        so, sig = occupancy_to_onsets(g)
        print(f"  scale {sc:>4}: onsets {[round(float(v), 2) for v in so[0]]}  "
              f"sigma {[round(float(v), 2) for v in sig[0]]}")
        assert torch.all(so[:, 1:] >= so[:, :-1] - 1e-4), "expected onsets must be monotone"
        assert abs(float(so[0, -1]) - T) < 1e-2, ("the row must still partition the clip", so)
        assert torch.isfinite(sig).all() and float(sig.min()) >= 0.0, sig
        if sc >= 50.0:
            # scale -> inf collapses the occupancy onto the Viterbi path, so the expectation
            # must collapse onto that path's integer onsets.
            assert float((so - hard_on.float()).abs().max()) < 0.01, (so, hard_on)
    print("  scale 50 matches the Viterbi onsets; float64 moments keep sigma finite")

    print("\n--- a fractional boundary reaches the target, and the crossing stays at 0.5 ---")
    # Planted boundary at 5.9 (onset 6.4): the target's own crossing must land there.
    on_f = torch.tensor([[0.0, 6.4, 20.0, 20.0, 20.0]])
    pif = onsets_to_phone_idx(on_f, torch.tensor([2]), 20)
    wf, pf = peak_weights(pif, on_f, torch.tensor([2]), ramp_frames=3.0)
    q0 = torch.where(pif[0] == 0, wf[0], 1 - wf[0])          # target mass on phone 0
    d = (1 - q0) - q0                                        # crosses zero at the boundary
    k = int(torch.nonzero(d[:-1] * d[1:] < 0)[0])
    cross = k + float(d[k] / (d[k] - d[k + 1]))
    print(f"  target crossing at frame {cross:.3f} (planted 5.900)")
    assert abs(cross - 5.9) < 1e-3, cross
    # Per-boundary ramps: the two phones meeting at a boundary use the SAME width there, so a
    # widened ramp softens the target without moving the crossing. Both phones must be at
    # least 2 * ramp long, otherwise `half` clips one side and only one side widens - which is
    # the asymmetry `soft_onset_ramp_max` exists to bound.
    def _crossing(idx, ons, ramp):
        wv, _ = peak_weights(idx, ons, torch.tensor([2]), ramp_frames=ramp)
        q = torch.where(idx[0] == 0, wv[0], 1 - wv[0])
        dd = (1 - q) - q
        kk = int(torch.nonzero(dd[:-1] * dd[1:] < 0)[0])
        return kk + float(dd[kk] / (dd[kk] - dd[kk + 1])), wv

    on_w = torch.tensor([[0.0, 16.4, 34.0, 34.0, 34.0]])     # both phones >= 14 frames
    pw = onsets_to_phone_idx(on_w, torch.tensor([2]), 34)
    c_narrow, w_narrow = _crossing(pw, on_w, 3.0)
    c_wide, w_wide = _crossing(pw, on_w, torch.tensor([[3.0, 7.0, 3.0, 3.0, 3.0]]))
    print(f"  crossing at ramp 3: {c_narrow:.3f}   at ramp 7: {c_wide:.3f}  (planted 15.900)")
    assert abs(c_narrow - 15.9) < 1e-3 and abs(c_wide - 15.9) < 1e-3, (c_narrow, c_wide)
    assert float(w_wide[0, 13]) < float(w_narrow[0, 13]), "a wider ramp must soften the target"
    print("  a per-boundary ramp widens the target without moving the crossing")

    print("\n--- loss decreases towards its own target ---")
    loss_fn = PeakLoss(min_frames=2, ramp_frames=3.0, group_weight=0.3)
    logits = torch.zeros(1, T, C, requires_grad=True)
    glogits = torch.zeros(1, T, 5, requires_grad=True)
    opt = torch.optim.Adam([logits, glogits], lr=0.2)
    first = last = None
    for step in range(120):
        lp = F.log_softmax(logits, dim=-1)
        glp = F.log_softmax(glogits, dim=-1)
        loss, stats, _ = loss_fn(lp, glp, seq, grp, torch.tensor([S]), torch.tensor([T]),
                                 prior_weight=1.0 if step < 20 else 0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    print(f"loss {first:.4f} -> {last:.4f}   kl={float(stats['peak_kl']):.4f} "
          f"entropy={float(stats['target_entropy']):.4f} "
          f"mean_phone={float(stats['mean_phone_frames']):.1f} frames")
    assert last < first - 0.3, (first, last)

    probs = F.log_softmax(logits, dim=-1).exp()[0]
    conf = probs.max(-1).values
    print("confidence trace:", " ".join(f"{float(v):.2f}" for v in conf[:24]))
    dips = peak_pick(confidence_dip(F.log_softmax(logits, dim=-1)[0]), k=3, min_distance=2)
    print("dips found at frames:", dips)
    assert len(dips) == 3

    print("\n--- ragged batch (different T and S, -100 padded, as the collator emits) ---")
    B, T, C = 4, 50, 12
    lens = torch.tensor([50, 37, 21, 8])
    slens = torch.tensor([6, 4, 3, 5])
    seqs = torch.randint(1, C - 1, (B, int(slens.max())))
    grps = torch.randint(1, 5, (B, int(slens.max())))
    pad = torch.arange(int(slens.max()))[None, :] >= slens[:, None]
    seqs = seqs.masked_fill(pad, -100)       # the real padding value; gathering it is a
    grps = grps.masked_fill(pad, -100)       # CUDA device-side assert if it is not clamped
    lp = F.log_softmax(torch.randn(B, T, C), dim=-1)
    glp = F.log_softmax(torch.randn(B, T, 5), dim=-1)
    loss, stats, al = loss_fn(lp, glp, seqs, grps, slens, lens)
    on = al['onsets']
    for b in range(B):
        row = on[b, :int(al['ext_lens'][b]) + 1].tolist()
        print(f"  clip {b}: T={int(lens[b]):>2} S={int(al['ext_lens'][b])} "
              f"feasible={bool(al['feasible'][b])} onsets={row}")
        assert row == sorted(row), row
        assert row[0] == 0 and row[-1] == int(lens[b]), row
    # The 8-frame clip holds 5+2 phones at 2 frames each: it cannot fit and must be dropped
    # from the loss rather than aligned wrongly.
    assert not bool(al['feasible'][3]) and bool(al['feasible'][0])
    assert torch.isfinite(loss), loss
    print(f"  loss={float(loss):.4f} infeasible_frac={float(stats['infeasible_frac']):.2f}")

    print("\n--- soft onsets in the loss ---")
    # The regression gate: at soft_onset_mix 0 the new code path must reduce to the old one
    # EXACTLY, not approximately, or none of the A/B numbers mean anything.
    base = PeakLoss(min_frames=2, ramp_frames=3.0, group_weight=0.3)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)
    lp_r = F.log_softmax(torch.randn(B, T, C, generator=g1), dim=-1)
    glp_r = F.log_softmax(torch.randn(B, T, 5, generator=g2), dim=-1)
    l0, s0, _ = base(lp_r, glp_r, seqs, grps, slens, lens)
    assert float(s0['target_entropy_ratio']) == 1.0, s0['target_entropy_ratio']
    print(f"  mix 0 (the old path): loss={float(l0):.6f}  "
          f"entropy_ratio={float(s0['target_entropy_ratio']):.3f}  "
          f"ramp_frac={float(s0['ramp_frac']):.3f}")

    soft = PeakLoss(min_frames=2, ramp_frames=3.0, group_weight=0.3,
                    soft_onset_mix=1.0, soft_onset_ramp_k=1.0, soft_onset_ramp_max=6.0)
    l2, s2, _ = soft(lp_r, glp_r, seqs, grps, slens, lens)
    print(f"  mix 1 + ramp k=1: loss={float(l2):.4f}  "
          f"entropy_ratio={float(s2['target_entropy_ratio']):.3f}  "
          f"shift={float(s2['onset_shift_frames']):.2f} frames  "
          f"sigma={float(s2['onset_sigma_frames']):.2f} frames")
    assert torch.isfinite(l2), l2
    # The whole point: whatever the occupancy does, the target stays a hat, so its entropy is
    # geometric and the ratio is pinned at 1. That is exactly what `soft_target_mix: 1.0` gave
    # up, and the reason the alignment drifted.
    assert abs(float(s2['target_entropy_ratio']) - 1.0) < 1e-5, s2['target_entropy_ratio']
    assert float(s2['onset_sigma_frames']) >= 0.0

    hard = PeakLoss(min_frames=2, ramp_frames=3.0, group_weight=0.3, soft_target_mix=1.0)
    _, sh, _ = hard(lp_r, glp_r, seqs, grps, slens, lens)
    print(f"  soft_target_mix 1 for contrast: entropy_ratio="
          f"{float(sh['target_entropy_ratio']):.3f} (the target the hat is measured against)")

    try:
        PeakLoss(soft_target_mix=0.5, soft_onset_mix=0.5)(
            lp_r, glp_r, seqs, grps, slens, lens)
        raise AssertionError("using the occupancy two ways at once must raise")
    except ValueError as e:
        print(f"  both mixes on -> ValueError: {str(e)[:50]}...")

    print("\nloss.py self-test OK")
