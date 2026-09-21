'''
WHY the boundaries miss: sort every TIMIT boundary a checkpoint gets wrong into a cause.

    /root/anaconda3/bin/python p3cupe/miss_breakdown.py <ckpt> [--bonus-weight W]

`error_breakdown.py` says WHERE the gap is (vowel onsets, laterals, rhotics, glides). This says
WHY. It matches p3's aligned phone sequence to TIMIT's phone-by-phone (edit distance on broad
classes, time as a tie-break) instead of boundary-by-boundary, so every TIMIT boundary lands in
exactly one bucket:

| bucket      | meaning                                                   | lever |
|-------------|-----------------------------------------------------------|-------|
| hit         | matched phone, onset within 25 ms                         | - |
| near        | matched, 25-50 ms                                         | precision: posteriors / refinement |
| far_run     | matched, >50 ms, and a matched neighbour is >50 ms off    | the path drifted: decoder |
|             | the same way                                              | (duration model, band) |
| far_single  | matched, >50 ms, neighbours fine                          | local confusion: the model |
| wrong_class | TIMIT phone aligned to a p3 phone of another broad class  | sequence/pronunciation mismatch |
| no_match    | TIMIT phone has no counterpart in the phonemizer sequence | sequence mismatch: unfixable by |
|             |                                                           | the aligner |

For every miss it also records whether the decoder's duration floor was binding: the p3
phone (or the one before it) sat at exactly `min_phone_frames` (the hard floor), or the TIMIT
phone itself is shorter than that floor (no path can place it). TIMIT closures are merged with
their release (`tcl t` -> one stop starting at the closure) because the phonemizer has one
phone for both - the stop onset convention question is `error_breakdown.py`'s, not this one's.

Onsets are the frame-grid Viterbi onsets (the probe's plain readout), not the sub-frame ones.
No training data is touched: this reads the TIMIT probe clips only.
'''

import argparse
import collections
import json
import os
import sys
from argparse import Namespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from p3cupe.error_breakdown import PATHS_LIST, PHONE_CLASS, read_phn
from p3cupe.timit_probe import DEFAULT_CACHE, _load_audio, load_cache

# Broad classes both sides can be mapped onto.
TIMIT_BROAD = {'vowel': 'V', 'rhotic': 'R', 'lateral': 'L', 'glide': 'G', 'nasal': 'N',
               'stop': 'S', 'closure': 'S', 'flap': 'S', 'fricative': 'F', 'h': 'F',
               'affricate': 'A', 'silence': '_'}
TIMIT_FIX = {'er': 'R', 'axr': 'R', 'nx': 'N'}
# ph66 pg16 groups -> broad class; the r-coloured schwa and the tap are special-cased.
PG16_BROAD = {0: '_', 1: 'V', 2: 'V', 3: 'V', 4: 'V', 5: 'V', 6: 'S', 7: 'S', 8: 'F', 9: 'F',
              10: 'A', 11: 'A', 12: 'N', 13: 'L', 14: 'R', 15: 'G', 16: '_'}
PH66_FIX = {9: 'R', 58: 'S'}           # ɚ is rhotic in TIMIT's terms; ɾ is TIMIT's flap dx
NAMES = {'V': 'vowel', 'R': 'rhotic', 'L': 'lateral', 'G': 'glide', 'N': 'nasal', 'S': 'stop',
         'F': 'fricative', 'A': 'affricate'}
NEAR = {frozenset(p) for p in ('VR', 'VG', 'VL', 'RL', 'NL', 'SA', 'FA', 'SF', 'RG', 'LG')}
REDUCED = {'ax', 'ix', 'ax-h', 'axr'}
BUCKETS = ('hit', 'near', 'far_run', 'far_single', 'wrong_class', 'no_match')


def p3_broad(ph):
    from utils.ph66_mapper import phoneme_groups_mapper
    return PH66_FIX.get(ph, PG16_BROAD[phoneme_groups_mapper[ph]])


def timit_tokens(phn):
    '''TIMIT segments -> tokens (broad, onset_ms, dur_ms, label), a closure merged into its release.'''
    out, k = [], 0
    while k < len(phn):
        a, b, lab = phn[k]
        cls = PHONE_CLASS.get(lab, '?')
        if cls == 'closure' and k + 1 < len(phn) and PHONE_CLASS.get(phn[k + 1][2]) in ('stop', 'affricate'):
            nxt = phn[k + 1]
            out.append(('A' if PHONE_CLASS[nxt[2]] == 'affricate' else 'S', a, nxt[1] - a,
                        f"{lab}+{nxt[2]}"))
            k += 2
            continue
        out.append((TIMIT_FIX.get(lab, TIMIT_BROAD.get(cls, '?')), a, b - a, lab))
        k += 1
    return out


def match(ref, hyp):
    '''Monotonic edit-distance alignment of token lists; returns {ref_idx: hyp_idx} for aligned pairs.'''
    def sub(r, h):
        c = 0.0 if r[0] == h[0] else (1.0 if frozenset((r[0], h[0])) in NEAR else 2.0)
        return c + 0.01 * min(abs(r[1] - h[1]), 150.0)

    def indel(t):
        return 0.3 if t[0] == '_' else 1.0

    n, m = len(ref), len(hyp)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0], bt[i][0] = D[i - 1][0] + indel(ref[i - 1]), 1
    for j in range(1, m + 1):
        D[0][j], bt[0][j] = D[0][j - 1] + indel(hyp[j - 1]), 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j], bt[i][j] = min((D[i - 1][j - 1] + sub(ref[i - 1], hyp[j - 1]), 0),
                                    (D[i - 1][j] + indel(ref[i - 1]), 1),
                                    (D[i][j - 1] + indel(hyp[j - 1]), 2))
    pairs, i, j = {}, n, m
    while i > 0 or j > 0:
        move = bt[i][j]
        if move == 0:
            pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return pairs


def mfa_tokens(path, ms):
    '''MFA's phones for the same clip, in run_model's format (reference only, never training).'''
    from p3cupe.error_breakdown import MFA_SILENCE
    with open(path) as f:
        seg = json.load(f)['segments'][0]
    out = []
    for p in seg['phoneme_ts']:
        lab = p['phone'].lower().rstrip('012')
        cls = '_' if lab in MFA_SILENCE else TIMIT_FIX.get(lab, TIMIT_BROAD.get(PHONE_CLASS.get(lab), '?'))
        out.append((cls, float(p['start_ms']), (p['end_ms'] - p['start_ms']) / ms, lab))
    return out


def run_model(pl_module, clips, batch_size=16):
    '''Per clip: list of used p3 phones as (broad, onset_ms, dur_frames, ph66).'''
    device = next(pl_module.parameters()).device
    ms = pl_module.ms_per_frame
    need_flux = bool(pl_module.loss_fn.bonus_weight)
    out = []
    for start in range(0, len(clips), batch_size):
        chunk = clips[start:start + batch_size]
        wavs = [_load_audio(c['audio'], pl_module.sample_rate) for c in chunk]
        wav_lens = torch.tensor([w.numel() for w in wavs])
        padded = torch.zeros(len(wavs), 1, int(wav_lens.max()))
        for i, w in enumerate(wavs):
            padded[i, 0, :w.numel()] = w
        padded = padded.to(device)
        spectral_lens = pl_module.calc_spec_len(wav_lens.to(device))
        with torch.no_grad():
            logits_c, _, _, flux = pl_module.encode_clip(
                padded, max_frames=int(spectral_lens.max()), return_flux=True)
            lp = F.log_softmax(logits_c.float(), dim=2)
            spectral_lens = spectral_lens.clamp(max=lp.shape[1])
            seqs = [torch.tensor(c['ph']) for c in chunk]
            seq_lens = torch.tensor([s.numel() for s in seqs])
            padded_seq = torch.zeros(len(seqs), int(seq_lens.max()), dtype=torch.long)
            for i, s in enumerate(seqs):
                padded_seq[i, :s.numel()] = s
            details = pl_module.align_batch(lp, padded_seq.to(device), seq_lens.to(device),
                                            spectral_lens, flux=flux if need_flux else None)
        for b, d in enumerate(details):
            if not d:
                out.append(None)
                continue
            n = int(spectral_lens[b])
            ends = d['onsets'][1:] + [n]
            out.append([(p3_broad(ph), on * ms, end - on, ph)
                        for ph, on, end, used in zip(d['phones'], d['onsets'], ends, d['used'])
                        if used])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--bonus-weight', type=float, default=None)
    ap.add_argument('--mfa', action='store_true',
                    help="score MFA's alignments instead of the checkpoint's (the reference row)")
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    from p3cupe.trainer import ComposedModel
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    hp = ck['hyper_parameters']
    model = ComposedModel(Namespace(**hp) if isinstance(hp, dict) else hp)
    model.load_state_dict(ck['state_dict'], strict=False)
    model = model.to(args.device).eval()
    model.fixed_window_batch = False
    if args.bonus_weight is not None:
        model.loss_fn.bonus_weight = args.bonus_weight
    ms = model.ms_per_frame
    d_min, d_soft = model.loss_fn.min_frames, model.loss_fn.soft_frames
    floor_ms = d_min * ms

    clips = load_cache(DEFAULT_CACHE)
    with open(PATHS_LIST) as f:
        by_id = {e['clip_id']: e for e in json.load(f)}
    if args.mfa:
        hyps = []
        for c in clips:
            e = by_id.get(c['clip_id'])
            path = e['vs'].replace('.vs.*', '.mfa.json') if e else None
            hyps.append(mfa_tokens(path, ms) if path and os.path.exists(path) else None)
    else:
        hyps = run_model(model, clips)

    buckets = collections.defaultdict(collections.Counter)      # class -> bucket -> n
    floor = collections.defaultdict(collections.Counter)        # class -> bucket -> n at floor
    too_short = collections.Counter()                           # class -> TIMIT dur < floor
    vowel_detail = collections.defaultdict(collections.Counter)  # (reduced?, left) -> bucket
    far_sign = collections.Counter()
    clip_far = []
    n_clips = n_infeasible = 0
    for clip, hyp in zip(clips, hyps):
        entry = by_id.get(clip['clip_id'])
        if entry is None:
            continue
        if hyp is None:
            n_infeasible += 1
            continue
        n_clips += 1
        ref = timit_tokens(read_phn(entry['PHN']))
        pairs = match(ref, hyp)
        errs = {i: hyp[j][1] - ref[i][1] for i, j in pairs.items() if hyp[j][0] == ref[i][0]}
        n_far = n_scored = 0
        for i in range(1, len(ref)):
            cls, onset, dur, lab = ref[i]
            if cls in ('_', '?'):
                continue
            n_scored += 1
            if dur < floor_ms:
                too_short[cls] += 1
            j = pairs.get(i)
            if j is None:
                b = 'no_match'
            elif hyp[j][0] != cls:
                b = 'wrong_class'
            else:
                e = errs[i]
                if abs(e) <= 25:
                    b = 'hit'
                elif abs(e) <= 50:
                    b = 'near'
                else:
                    run = any(k in errs and abs(errs[k]) > 50 and (errs[k] > 0) == (e > 0)
                              for k in (i - 1, i + 1))
                    b = 'far_run' if run else 'far_single'
                    far_sign['late' if e > 0 else 'early'] += 1
                    n_far += 1
            buckets[cls][b] += 1
            if j is not None and b != 'hit':
                at_floor = hyp[j][2] <= d_min or (j > 0 and hyp[j - 1][2] <= d_min)
                floor[cls][b] += int(at_floor)
            if cls == 'V':
                left = ref[i - 1][0] if ref[i - 1][0] != '_' else 'sil'
                vowel_detail[('reduced' if lab in REDUCED else 'full', NAMES.get(left, left))][b] += 1
        clip_far.append(n_far / max(n_scored, 1))

    print(f"\n{'MFA (reference)' if args.mfa else 'p3'}: {n_clips} clips ({n_infeasible} infeasible skipped), epoch {ck.get('epoch')}, "
          f"bonus {model.loss_fn.bonus_weight:g}, floor {d_min} frames = {floor_ms:.0f} ms "
          f"(+{d_soft} soft)")

    def pct(c, k):
        return 100.0 * c[k] / max(sum(c.values()), 1)

    print(f"\nBuckets per TIMIT phone class (% of that class's boundaries); "
          f"'floor' = % of the non-hit ones where p3's phone or its left neighbour sat at the "
          f"{d_min}-frame floor")
    print(f"  {'class':10s} {'n':>5s} | " + " ".join(f"{b:>11s}" for b in BUCKETS)
          + f" | {'floor':>5s} {'<floor':>6s}")
    total, total_floor, total_short = collections.Counter(), collections.Counter(), 0
    for cls in sorted(buckets, key=lambda c: -sum(buckets[c].values())):
        c = buckets[cls]
        n = sum(c.values())
        misses = n - c['hit'] - c['no_match']
        total.update(c)
        total_floor.update(floor[cls])
        total_short += too_short[cls]
        print(f"  {NAMES.get(cls, cls):10s} {n:5d} | " + " ".join(f"{pct(c, b):11.1f}" for b in BUCKETS)
              + f" | {100.0 * sum(floor[cls].values()) / max(misses, 1):5.1f} "
                f"{100.0 * too_short[cls] / n:6.1f}")
    n = sum(total.values())
    misses = n - total['hit'] - total['no_match']
    print(f"  {'ALL':10s} {n:5d} | " + " ".join(f"{pct(total, b):11.1f}" for b in BUCKETS)
          + f" | {100.0 * sum(total_floor.values()) / max(misses, 1):5.1f} "
            f"{100.0 * total_short / n:6.1f}")
    print(f"\n  far misses: {far_sign['early']} early / {far_sign['late']} late")

    print("\nVowel onsets by vowel type and left neighbour (% of row)")
    print(f"  {'':22s} {'n':>5s} | " + " ".join(f"{b:>11s}" for b in BUCKETS))
    for key in sorted(vowel_detail, key=lambda k: -sum(vowel_detail[k].values())):
        c = vowel_detail[key]
        if sum(c.values()) < 60:
            continue
        print(f"  {key[0] + ' after ' + key[1]:22s} {sum(c.values()):5d} | "
              + " ".join(f"{pct(c, b):11.1f}" for b in BUCKETS))

    clip_far.sort()
    q = lambda p: clip_far[int(p * (len(clip_far) - 1))]
    worst = sum(clip_far[-len(clip_far) // 10:]) / max(sum(clip_far), 1e-9)
    print(f"\nPer-clip share of boundaries >50 ms off: median {100 * q(0.5):.1f}%, "
          f"p90 {100 * q(0.9):.1f}%, max {100 * clip_far[-1]:.1f}%; "
          f"the worst 10% of clips hold {100 * worst:.0f}% of all far misses")


if __name__ == "__main__":
    main()
