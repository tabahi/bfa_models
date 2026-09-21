'''
Where the boundary errors are: a p3cupe checkpoint against MFA, per TIMIT phone class.

    /root/anaconda3/bin/python modules/p3cupe/error_breakdown.py <ckpt> [--half holdout|tune|all]

Both aligners are scored on the SAME probe clips with the probe's own scorer
(`levenshtein_align`, onsets in ms), and every labelled boundary is attributed to the TIMIT
phones on either side of it (read from the clip's `.PHN`). MFA's per-file outputs are the
`.mfa.json` files behind `results_TIMIT_lev.csv`'s MFA row.

A boundary between a stop's closure and its burst (`tcl` -> `t`) exists in TIMIT's labels but
not in any phonemizer sequence: one phone `t` covers both. Neither aligner can place it except
by accident, so those are reported as their own row rather than blended into "stops".
'''

import argparse
import collections
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_paths import get_path
from utils.levenshtein import levenshtein_align
from p3cupe.timit_probe import DEFAULT_CACHE, _load_audio, _onsets_ms, load_cache

PATHS_LIST = get_path("timit_paths_list")
MFA_SILENCE = {'', 'sil', 'sp', 'spn'}

CLASSES = {
    'closure': 'bcl dcl gcl pcl tcl kcl',
    'stop': 'b d g p t k q',
    'flap': 'dx nx',
    'affricate': 'jh ch',
    'fricative': 's sh z zh f th v dh',
    'h': 'hh hv',
    'nasal': 'm n ng em en eng',
    'lateral': 'l el',
    'rhotic': 'r er axr',
    'glide': 'w y',
    'vowel': 'iy ih eh ey ae aa aw ay ah ao oy ow uh uw ux ax ix ax-h',
    'silence': 'h# pau epi',
}
PHONE_CLASS = {p: c for c, ps in CLASSES.items() for p in ps.split()}


def read_phn(path):
    with open(path) as f:
        rows = [line.split() for line in f if line.strip()]
    return [(int(a) / 16.0, int(b) / 16.0, lab) for a, b, lab in rows]


def mfa_onsets(path):
    with open(path) as f:
        seg = json.load(f)['segments'][0]
    return sorted({int(round(p['start_ms'])) for p in seg['phoneme_ts']
                   if p['phone'].lower() not in MFA_SILENCE and p['start_ms'] > 0})


def p3_onsets(pl_module, clips, batch_size=16):
    '''Onsets in ms per clip from the checkpoint's own aligner, exactly as the probe reads them.'''
    device = next(pl_module.parameters()).device
    out = []
    need_flux = bool(pl_module.loss_fn.bonus_weight)
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
        for d in details:
            if not d:
                out.append([])
                continue
            out.append(_onsets_ms([(ph, raw * pl_module.ms_per_frame) for ph, raw, used
                                   in zip(d['phones'], d['onsets'], d['used']) if used]))
    return out


class Tally:
    def __init__(self):
        self.n = 0
        self.hit = collections.Counter()
        self.abs_err = []
        self.signed = []

    def add(self, err):
        self.n += 1
        if err is None:
            return
        self.abs_err.append(abs(err))
        self.signed.append(err)
        for t in (10, 25, 50):
            if abs(err) <= t:
                self.hit[t] += 1

    def row(self):
        r = {t: 100.0 * self.hit[t] / max(self.n, 1) for t in (10, 25, 50)}
        m = sorted(self.abs_err)
        r['med'] = m[len(m) // 2] if m else float('nan')
        s = sorted(self.signed)
        r['bias'] = s[len(s) // 2] if s else float('nan')
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--half', default='all', choices=('all', 'holdout', 'tune'))
    ap.add_argument('--bonus-weight', type=float, default=None,
                    help="override boundary_bonus_weight (0 = the model's own alignment)")
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    from argparse import Namespace
    from p3cupe.trainer import ComposedModel
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    hp = ck['hyper_parameters']
    hp = Namespace(**hp) if isinstance(hp, dict) else hp
    model = ComposedModel(hp)
    model.load_state_dict(ck['state_dict'], strict=False)
    model = model.to(args.device).eval()
    model.fixed_window_batch = False
    if args.bonus_weight is not None:
        model.loss_fn.bonus_weight = args.bonus_weight

    clips = load_cache(DEFAULT_CACHE)
    if args.half != 'all':
        clips = [c for i, c in enumerate(clips) if (i % 2 == 0) == (args.half == 'tune')]
    with open(PATHS_LIST) as f:
        by_id = {e['clip_id']: e for e in json.load(f)}

    ours = p3_onsets(model, clips)
    by_class = {m: collections.defaultdict(Tally) for m in ('p3', 'mfa')}
    by_pair = {m: collections.defaultdict(Tally) for m in ('p3', 'mfa')}
    totals = {m: [0, 0, 0] for m in ('p3', 'mfa')}              # hits@25, n_pred, n_known
    n_clips = 0
    for clip, pred_p3 in zip(clips, ours):
        entry = by_id.get(clip['clip_id'])
        if entry is None:
            continue
        mfa_path = entry['vs'].replace('.vs.*', '.mfa.json')
        if not os.path.exists(mfa_path):
            continue
        phn = read_phn(entry['PHN'])
        # Labelled boundaries = onsets of every PHN segment after the first, as in the cache.
        known = [int(round(a)) for a, _, _ in phn[1:]]
        sides = [(phn[k - 1][2], phn[k][2]) for k in range(1, len(phn))]
        n_clips += 1
        for method, pred in (('p3', pred_p3), ('mfa', mfa_onsets(mfa_path))):
            err = {ki: e for ki, _, e in levenshtein_align(known, pred, max_dist=100.0)}
            totals[method][1] += len(pred)
            totals[method][2] += len(known)
            for k, (left, right) in enumerate(sides):
                lc, rc = PHONE_CLASS.get(left, '?'), PHONE_CLASS.get(right, '?')
                e = err.get(k)
                if e is not None and abs(e) <= 25:
                    totals[method][0] += 1
                key = 'closure->burst' if lc == 'closure' and rc == 'stop' else rc
                by_class[method][key].add(e)
                by_pair[method][(lc, rc)].add(e)

    print(f"\n{n_clips} clips ({args.half}), checkpoint epoch {ck.get('epoch')}, "
          f"bonus weight {model.loss_fn.bonus_weight:g}")
    for m in ('p3', 'mfa'):
        h, p, k = totals[m]
        pr, rc = h / max(p, 1), h / max(k, 1)
        print(f"  {m:4s} f1@25 {200 * pr * rc / max(pr + rc, 1e-9):5.1f}  (recall {100 * rc:.1f}, "
              f"precision {100 * pr:.1f}, {p} predicted / {k} labelled)")

    def table(groups, title, min_n=40, top=None):
        keys = sorted(groups['mfa'], key=lambda g: -groups['mfa'][g].n)
        keys = [g for g in keys if groups['mfa'][g].n >= min_n][:top]
        print(f"\n{title}: recall within 10/25/50 ms, median |err|, median signed err (ms)")
        print(f"  {'':24s} {'n':>5s} | {'p3 @10':>6s} {'@25':>5s} {'@50':>5s} {'med':>5s} "
              f"{'bias':>5s} | {'mfa@10':>6s} {'@25':>5s} {'@50':>5s} {'med':>5s} {'bias':>5s} | "
              f"{'Δ@25':>5s}")
        for g in keys:
            a, b = groups['p3'][g].row(), groups['mfa'][g].row()
            name = g if isinstance(g, str) else f"{g[0]} -> {g[1]}"
            print(f"  {name:24s} {groups['mfa'][g].n:5d} | {a[10]:6.1f} {a[25]:5.1f} {a[50]:5.1f} "
                  f"{a['med']:5.1f} {a['bias']:+5.1f} | {b[10]:6.1f} {b[25]:5.1f} {b[50]:5.1f} "
                  f"{b['med']:5.1f} {b['bias']:+5.1f} | {a[25] - b[25]:+5.1f}")

    table(by_class, "By the phone that STARTS at the boundary")
    table(by_pair, "Most frequent transitions", min_n=60, top=25)


if __name__ == "__main__":
    main()
