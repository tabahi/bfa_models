'''
Decoder-only sweep: re-align one checkpoint's posteriors under different decoder settings.

    /root/anaconda3/bin/python modules/p3cupe/decoder_sweep.py <ckpt> [--half tune|holdout|all]
        [--bonus 0 3 10 30] [--min 4 5 6] [--soft 4]

The model runs once per clip; only `viterbi_align` is repeated, so a grid takes minutes. Each
setting is scored two ways:
  - `phone@25`: per-phone onset hit rate (miss_breakdown.py's matching; the stop onset is
    at the closure);
  - `lev f1@25`: the probe's boundary scorer on the frame grid, which also credits TIMIT's
    closure->burst boundaries.
Sweep on the TUNE half; confirm the pick on the holdout half.
'''

import argparse
import collections
import itertools
import json
import os
import sys
from argparse import Namespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.levenshtein import levenshtein_align
from p3cupe.error_breakdown import PATHS_LIST, read_phn
from p3cupe.miss_breakdown import match, p3_broad, timit_tokens
from p3cupe.timit_probe import DEFAULT_CACHE, _load_audio, _onsets_ms, load_cache

CLASSES = ('V', 'R', 'L', 'G', 'N', 'S', 'F')


@torch.no_grad()
def encode_all(pl_module, clips, batch_size=16):
    device = next(pl_module.parameters()).device
    batches = []
    for start in range(0, len(clips), batch_size):
        chunk = clips[start:start + batch_size]
        wavs = [_load_audio(c['audio'], pl_module.sample_rate) for c in chunk]
        wav_lens = torch.tensor([w.numel() for w in wavs])
        padded = torch.zeros(len(wavs), 1, int(wav_lens.max()))
        for i, w in enumerate(wavs):
            padded[i, 0, :w.numel()] = w
        spectral_lens = pl_module.calc_spec_len(wav_lens.to(device))
        logits_c, _, _, flux = pl_module.encode_clip(
            padded.to(device), max_frames=int(spectral_lens.max()), return_flux=True)
        lp = F.log_softmax(logits_c.float(), dim=2)
        seqs = [torch.tensor(c['ph']) for c in chunk]
        seq_lens = torch.tensor([s.numel() for s in seqs])
        padded_seq = torch.zeros(len(seqs), int(seq_lens.max()), dtype=torch.long)
        for i, s in enumerate(seqs):
            padded_seq[i, :s.numel()] = s
        batches.append((lp, flux, padded_seq.to(device), seq_lens.to(device),
                        spectral_lens.clamp(max=lp.shape[1])))
    return batches


def expected_details(pl_module, lp, seq, seq_lens, spec_lens, flux, scale):
    '''align_batch-shaped output whose onsets are POSTERIOR-MEAN onsets at this path-score scale.

    Alignments are monotonic, so phone s starts after every frame spent in phones < s:
    E[onset_s] = sum_t P(phone at t < s) = sum_t cumsum_s(gamma)[t, s - 1].
    '''
    from p3cupe.loss import alignment_occupancy, boundary_bonus, prepare_targets
    lf = pl_module.loss_fn
    ext, _, ext_lens, oh, ot = prepare_targets(seq, seq, seq_lens, lf.sil_class, lf.optional_sil)
    bonus = boundary_bonus(flux, spec_lens, lf.contrast_frames) if lf.bonus_weight else None
    gamma, feas = alignment_occupancy(
        lp, ext, ext_lens, spec_lens, min_frames=lf.min_frames, opt_head=oh, opt_tail=ot,
        class_prior_norm=lf.class_prior_norm, soft_frames=lf.soft_frames, soft_cost=lf.soft_cost,
        sil_skippable=lf.sil_skippable, sil_class=lf.sil_class, onset_bonus=bonus,
        bonus_weight=lf.bonus_weight, scale=scale)
    before = gamma.cumsum(-1).sum(1)                                        # [B, S]: E[frames in phones <= s]
    onsets = torch.cat([torch.zeros_like(before[:, :1]), before[:, :-1]], dim=1)
    out = []
    for b in range(lp.shape[0]):
        n = int(ext_lens[b])
        if not bool(feas[b]):
            out.append({})
            continue
        on = onsets[b, :n].tolist()
        dur = gamma[b].sum(0)[:n].tolist()
        out.append({'phones': ext[b, :n].tolist(), 'onsets': on, 'used': [d >= 0.5 for d in dur]})
    return out


@torch.no_grad()
def score(pl_module, batches, clips, refs, occupancy_scale=None):
    ms = pl_module.ms_per_frame
    hits, counts = collections.Counter(), collections.Counter()
    lev_hits = n_known = n_pred = 0
    clip_i = 0
    for lp, flux, seq, seq_lens, spec_lens in batches:
        if occupancy_scale:
            details = expected_details(pl_module, lp, seq, seq_lens, spec_lens, flux, occupancy_scale)
        else:
            details = pl_module.align_batch(lp, seq, seq_lens, spec_lens,
                                            flux=flux if pl_module.loss_fn.bonus_weight else None)
        for b, d in enumerate(details):
            clip, ref = clips[clip_i], refs[clip_i]
            clip_i += 1
            if not d or ref is None:
                continue
            used = [(ph, on) for ph, on, u in zip(d['phones'], d['onsets'], d['used']) if u]
            # timit_probe's readout, frame grid
            known, pred = clip['boundaries_ms'], _onsets_ms([(ph, on * ms) for ph, on in used])
            n_known, n_pred = n_known + len(known), n_pred + len(pred)
            lev_hits += sum(abs(e) <= 25 for _, _, e in levenshtein_align(known, pred, max_dist=100.0))
            # per phone
            hyp = [(p3_broad(ph), on * ms) for ph, on in used]
            pairs = match(ref, hyp)
            for i in range(1, len(ref)):
                cls = ref[i][0]
                if cls not in CLASSES:
                    continue
                counts[cls] += 1
                j = pairs.get(i)
                if j is not None and hyp[j][0] == cls and abs(hyp[j][1] - ref[i][1]) <= 25:
                    hits[cls] += 1
    pr, rc = lev_hits / max(n_pred, 1), lev_hits / max(n_known, 1)
    out = {'lev': 200 * pr * rc / max(pr + rc, 1e-9),
           'phone': 100 * sum(hits.values()) / max(sum(counts.values()), 1)}
    out.update({c: 100 * hits[c] / max(counts[c], 1) for c in CLASSES})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--half', default='tune', choices=('all', 'holdout', 'tune'))
    ap.add_argument('--bonus', type=float, nargs='+', default=[0, 3, 10, 30])
    ap.add_argument('--min', type=int, nargs='+', default=[4, 5, 6])
    ap.add_argument('--soft', type=int, nargs='+', default=[4])
    ap.add_argument('--occupancy-scale', type=float, nargs='+', default=[0],
                    help="0 = Viterbi onsets; >0 = posterior-mean onsets from forward-backward at "
                         "this path-score scale")
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    from p3cupe.trainer import ComposedModel
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    hp = ck['hyper_parameters']
    model = ComposedModel(Namespace(**hp) if isinstance(hp, dict) else hp)
    model.load_state_dict(ck['state_dict'], strict=False)
    model = model.to(args.device).eval()
    model.fixed_window_batch = False
    trained = (model.loss_fn.bonus_weight, model.loss_fn.min_frames, model.loss_fn.soft_frames)

    clips = load_cache(DEFAULT_CACHE)
    if args.half != 'all':
        clips = [c for i, c in enumerate(clips) if (i % 2 == 0) == (args.half == 'tune')]
    with open(PATHS_LIST) as f:
        by_id = {e['clip_id']: e for e in json.load(f)}
    refs = [timit_tokens(read_phn(by_id[c['clip_id']]['PHN'])) if c['clip_id'] in by_id else None
            for c in clips]
    batches = encode_all(model, clips)

    print(f"\n{len(clips)} clips ({args.half}), epoch {ck.get('epoch')}; trained with "
          f"bonus {trained[0]:g}, min {trained[1]}, soft {trained[2]}; frame {model.ms_per_frame:g} ms")
    print(f"  {'occ':>5s} {'bonus':>5s} {'min':>3s} {'soft':>4s} | {'phone@25':>8s} {'lev f1@25':>9s} | "
          + " ".join(f"{c:>5s}" for c in CLASSES))
    for occ, w, d, k in itertools.product(args.occupancy_scale, args.bonus, args.min, args.soft):
        model.loss_fn.bonus_weight, model.loss_fn.min_frames, model.loss_fn.soft_frames = w, d, k
        r = score(model, batches, clips, refs, occupancy_scale=occ)
        print(f"  {occ:5g} {w:5g} {d:3d} {k:4d} | {r['phone']:8.2f} {r['lev']:9.2f} | "
              + " ".join(f"{r[c]:5.1f}" for c in CLASSES), flush=True)


if __name__ == "__main__":
    main()
