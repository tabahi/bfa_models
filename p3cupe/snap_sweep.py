'''
Offline boundary snapping: move each forced-aligned onset to a nearby peak of representation
change, and score whether that helps. No training.

    /root/anaconda3/bin/python p3cupe/snap_sweep.py <ckpt> [--hp p3cupe/cupe_hp]
        [--taps fe ctx emb hid post mel] [--span 1 2] [--radius 1 2 3] [--min-z -99 0 1]

The question it answers: does the change in some layer carry boundary TIMING that the aligner
is missing? The aligner already fixes count, order and identity; snapping only moves each onset
within +-radius frames, so any gain is timing information alone. If snapping does not help, a
boundary head trained toward the same change signal will not help either (memory.md, "Ideas
discussed, not started").

## Knobs
- tap: which tensor the change is read from, in forward order. `fe` is the downsampler output
  (before the frequency gate, the aligner's `tap` flux), `ctx` is after the context convs,
  `emb` the final embedding (the probe's `flux_` readout), `hid` the classifier's hidden layer
  (the MLP), `post` the phone posteriors, and `mel` parameter-free log-mel (the onset bonus's
  source; it is known to drag stop onsets onto bursts).
- span k: change at t = cosine distance between the mean of frames [t-k, t) and [t, t+k).
  k=1 is exactly `loss.embedding_delta`. `mel` uses L2 instead of cosine.
- radius r: search +-r frames around the onset for a local maximum; the strongest wins and is
  parabolically refined.
- min-z: snap only when that peak's per-clip z-score clears this; otherwise leave the onset.

## Conventions, stated because they move the numbers
Onset frame `o` is the first frame of the new phone; the hat puts its 0.5 crossing at `o - 0.5`,
and a change-signal peak at index `i` means a boundary between frames i-1 and i, i.e. `i - 0.5`.
So snapped onsets are reported at `(i - 0.5) * ms`. The probe's frame-grid readouts report `o * ms`
- half a frame (2.5 ms) later - so BOTH baselines are printed: `grid o` reproduces the logged
`nobonus_f1_25ms`, `grid o-0.5` is the like-for-like reference for every snapped row.

Per-phone hit@25 uses `miss_breakdown.match` (a stop onset at its closure), as `decoder_sweep` does.
Sweep on the tune half, read the pick on the holdout half.
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
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.levenshtein import levenshtein_align
from p3cupe import windowing
from p3cupe.error_breakdown import PATHS_LIST, read_phn
from p3cupe.miss_breakdown import match, p3_broad, timit_tokens
from p3cupe.timit_probe import DEFAULT_CACHE, _load_audio, load_cache

CLASSES = ('V', 'R', 'L', 'G', 'N', 'S', 'F')
SIL = 0
TOL = 25.0


@torch.no_grad()
def forward_taps(pl, padded, spectral_lens):
    '''One windowed forward, returning log-posteriors and every tap, stitched to the clip grid.'''
    net = pl.model
    length = padded.shape[-1]
    n_win = windowing.num_windows_for(length, pl.window_size_wav, pl.stride_size_wav)
    t_full = windowing.total_stitched_frames(n_win, pl.frames_per_window, pl.stride_frames)
    windows = windowing.slice_windows(padded, pl.sample_rate, pl.window_size_ms, pl.stride_ms)
    b, n_win, win_len = windows.shape
    x = windows.reshape(-1, win_len)

    fe = net.feature_extractor(x.unsqueeze(1))
    h = net.context(fe * net.freq_attention(fe))
    ctx = h.transpose(1, 2)
    emb = net.norm(net.blocks(ctx))
    logits, _ = net.classify(emb)
    hid = net.classifier[1](net.classifier[0](emb))          # the MLP's hidden layer, post-GELU
    # (with local_delta the delta branch is not in `hid`; the tap is the embedding branch only)

    def stitch(t):
        out = windowing.stitch_windows(t.reshape(b, n_win, pl.frames_per_window, -1),
                                       num_windows_total=n_win,
                                       frames_per_window=pl.frames_per_window,
                                       stride_frames=pl.stride_frames)
        return F.pad(out, (0, 0, 0, max(0, t_full - out.shape[1])))

    lp = F.log_softmax(stitch(logits).float(), dim=-1)
    taps = {'fe': stitch(fe.transpose(1, 2)), 'ctx': stitch(ctx), 'emb': stitch(emb),
            'hid': stitch(hid), 'post': lp.exp()}
    taps['mel'] = pl._mel_flux(padded, t_full)                 # already a [B, T] change signal
    return lp, taps


def change_signal(x, k, l2=False):
    '''[T, D] -> [T]; entry t = change between the k frames before t and the k frames from t.'''
    x = x.float()
    t = x.shape[0]
    c = F.pad(x.cumsum(0), (0, 0, 1, 0))                       # c[i] = sum x[:i]
    idx = torch.arange(t, device=x.device)
    lo, hi = (idx - k).clamp(min=0), (idx + k).clamp(max=t)
    before = (c[idx] - c[lo]) / (idx - lo).clamp(min=1).unsqueeze(1)
    after = (c[hi] - c[idx]) / (hi - idx).clamp(min=1).unsqueeze(1)
    if l2:
        s = (after - before).norm(dim=-1)
    else:
        s = 1.0 - (F.normalize(before, dim=-1) * F.normalize(after, dim=-1)).sum(-1)
    s[0] = 0.0
    return s


def snap(onsets, sig, r, min_z):
    '''
    onsets: integer onset frames (strictly increasing). Returns boundary positions in FRAMES,
    on the `i - 0.5` convention, same length and still strictly increasing.
    '''
    x = sig.tolist()
    n = len(x)
    z = (sig - sig.mean()) / sig.std().clamp(min=1e-9)
    zl = z.tolist()
    out, prev = [], -1e9
    for o in onsets:
        best = None
        for i in range(max(1, o - r), min(n - 1, o + r + 1)):
            if x[i] >= x[i - 1] and x[i] > x[i + 1] and zl[i] >= min_z:
                if best is None or x[i] > x[best]:
                    best = i
        pos = o - 0.5
        if best is not None:
            a, bb, c = x[best - 1], x[best], x[best + 1]
            d = a - 2 * bb + c
            shift = 0.5 * (a - c) / d if abs(d) > 1e-12 else 0.0
            cand = best + max(-0.5, min(0.5, shift)) - 0.5
            if cand > prev + 1.0:          # never reorder or merge two phones
                pos = cand
        if pos <= prev + 1.0:
            pos = max(o - 0.5, prev + 1.0)
        out.append(pos)
        prev = pos
    return out


class Score:
    def __init__(self):
        self.hits = self.known = self.pred = 0
        self.err = []
        self.ph_hit, self.ph_n = collections.Counter(), collections.Counter()

    def add(self, known, phones_ms, ref):
        pred = sorted({int(round(ms)) for ph, ms in phones_ms if ph != SIL and ms > 0})
        if not pred or not known:
            return
        self.known += len(known)
        self.pred += len(pred)
        for _, _, e in levenshtein_align(known, pred, max_dist=100.0):
            self.err.append(abs(e))
            self.hits += abs(e) <= TOL
        if ref is None:
            return
        hyp = [(p3_broad(ph), ms) for ph, ms in phones_ms]
        pairs = match(ref, hyp)
        for i in range(1, len(ref)):
            cls = ref[i][0]
            if cls not in CLASSES:
                continue
            self.ph_n[cls] += 1
            j = pairs.get(i)
            if j is not None and hyp[j][0] == cls and abs(hyp[j][1] - ref[i][1]) <= TOL:
                self.ph_hit[cls] += 1

    def result(self):
        p, r = self.hits / max(self.pred, 1), self.hits / max(self.known, 1)
        return {'f1': 200 * p * r / max(p + r, 1e-9),
                'err': sum(self.err) / max(len(self.err), 1),
                'phone': 100 * sum(self.ph_hit.values()) / max(sum(self.ph_n.values()), 1),
                **{c: 100 * self.ph_hit[c] / max(self.ph_n[c], 1) for c in CLASSES}}


def load_model(ckpt_path, hp_path, device):
    from p3cupe.trainer import ComposedModel
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hp = ck['hyper_parameters']
    hp = Namespace(**hp) if isinstance(hp, dict) else hp
    if ck.get('model_config'):                                  # the shapes actually saved
        hp.local_delta = int(ck['model_config'].get('local_delta', 0))
    model = ComposedModel(hp)
    state = {k: v for k, v in ck['state_dict'].items() if not k.startswith('loss_fn')}
    missing, _ = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith('loss_fn')]
    if missing:
        raise ValueError(f"checkpoint is missing {missing[:6]}")
    # Decoder settings from the LIVE yaml, as the in-training probe uses them, so the unsnapped
    # baseline reproduces the logged number.
    if hp_path:
        with open(hp_path + '.yaml') as f:
            model.hp = Namespace(**{**vars(model.hp), **yaml.safe_load(f)})
        model._read_decoder_hp()
        lf = model.loss_fn
        lf.min_frames = int(getattr(model.hp, 'min_phone_frames', lf.min_frames))
        lf.optional_sil = bool(getattr(model.hp, 'optional_sil', lf.optional_sil))
        lf.class_prior_norm = bool(getattr(model.hp, 'class_prior_norm', lf.class_prior_norm))
    model.loss_fn.bonus_weight = 0.0                            # snapping replaces the bonus
    model.fixed_window_batch = False
    return model.to(device).eval(), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--hp', default='p3cupe/cupe_hp', help="yaml for decoder settings ('' = ckpt's)")
    ap.add_argument('--taps', nargs='+', default=['fe', 'ctx', 'emb', 'hid', 'post', 'mel'])
    ap.add_argument('--span', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--radius', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--min-z', type=float, nargs='+', default=[-99.0, 0.0, 1.0])
    ap.add_argument('--offset', type=float, default=0.0,
                    help="frames added to every snapped onset (and to `grid o-0.5`); 0.5 puts "
                         "them on the probe's `o` convention")
    ap.add_argument('--top', type=int, default=8, help="best tune configs to show on holdout")
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    model, ck = load_model(args.ckpt, args.hp, args.device)
    ms = model.ms_per_frame
    clips = load_cache(DEFAULT_CACHE)
    with open(PATHS_LIST) as f:
        by_id = {e['clip_id']: e for e in json.load(f)}
    refs = [timit_tokens(read_phn(by_id[c['clip_id']]['PHN'])) if c['clip_id'] in by_id else None
            for c in clips]

    configs = [('grid o', None, 0, 0, 0.0), ('grid o-0.5', None, 0, 0, 0.0)]
    configs += [(t, t, k, r, z) for t, k, r, z in
                itertools.product(args.taps, args.span, args.radius, args.min_z)]
    scores = {(h, c): Score() for h in ('tune', 'hold') for c in configs}

    for start in range(0, len(clips), 16):
        chunk = clips[start:start + 16]
        wavs = [_load_audio(c['audio'], model.sample_rate) for c in chunk]
        wav_lens = torch.tensor([w.numel() for w in wavs])
        padded = torch.zeros(len(wavs), 1, int(wav_lens.max()))
        for i, w in enumerate(wavs):
            padded[i, 0, :w.numel()] = w
        padded = padded.to(args.device)
        spec = model.calc_spec_len(wav_lens.to(args.device))
        lp, taps = forward_taps(model, padded, spec)
        spec = spec.clamp(max=lp.shape[1])
        seqs = [torch.tensor(c['ph']) for c in chunk]
        seq_lens = torch.tensor([s.numel() for s in seqs])
        pseq = torch.zeros(len(seqs), int(seq_lens.max()), dtype=torch.long)
        for i, s in enumerate(seqs):
            pseq[i, :s.numel()] = s
        details = model.align_batch(lp, pseq.to(args.device), seq_lens.to(args.device), spec)

        for i, d in enumerate(details):
            ci = start + i
            half = 'tune' if ci % 2 == 0 else 'hold'
            known = sorted(set(clips[ci]['boundaries_ms']))
            if not d or not known:
                continue
            used = [(ph, int(on)) for ph, on, u in zip(d['phones'], d['onsets'], d['used']) if u]
            phones = [ph for ph, _ in used]
            ons = [on for _, on in used]
            n = int(spec[i])
            sigs = {}
            for cfg in configs:
                name, tap, k, r, z = cfg
                if tap is None:
                    pos = [o * ms for o in ons] if name == 'grid o' else [(o - 0.5 + args.offset) * ms for o in ons]
                    pos[0] = ons[0] * ms if name == 'grid o' else max(0.0, (ons[0] - 0.5) * ms)
                else:
                    key = (tap, k)
                    if key not in sigs:
                        sigs[key] = (taps['mel'][i, :n].float() if tap == 'mel' and k == 1 else
                                     change_signal(taps['mel'][i, :n, None], k, l2=True) if tap == 'mel'
                                     else change_signal(taps[tap][i, :n], k)).cpu()
                    # the first phone's onset is not a boundary; leave it where it is
                    pos = [ons[0] - 0.5] + snap(ons[1:], sigs[key], r, z)
                    pos = [max(0.0, p + args.offset) * ms for p in pos]
                scores[(half, cfg)].add(known, list(zip(phones, pos)), refs[ci])

    res = {key: s.result() for key, s in scores.items()}
    print(f"\n{os.path.basename(args.ckpt)}  (epoch {ck.get('epoch')}); aligner bonus OFF; "
          f"{len(clips)} clips; f1@25 = probe scorer, phone@25 = per-phone hit rate")
    head = (f"  {'config':<24s} | {'tune f1':>7s} {'phone':>6s} | {'hold f1':>7s} {'phone':>6s} "
            f"{'err':>5s} | " + " ".join(f"{c:>5s}" for c in CLASSES))

    def row(cfg):
        name, tap, k, r, z = cfg
        label = name if tap is None else f"{tap} k{k} r{r} z{z:g}"
        t, h = res[('tune', cfg)], res[('hold', cfg)]
        return (f"  {label:<24s} | {t['f1']:7.2f} {t['phone']:6.2f} | {h['f1']:7.2f} {h['phone']:6.2f} "
                f"{h['err']:5.1f} | " + " ".join(f"{h[c]:5.1f}" for c in CLASSES))

    print(head)
    for cfg in configs[:2]:
        print(row(cfg))
    print("  -- best per tap on the TUNE half (by phone@25), holdout alongside --")
    for tap in args.taps:
        cands = [c for c in configs[2:] if c[1] == tap]
        print(row(max(cands, key=lambda c: res[('tune', c)]['phone'])))
    print(f"  -- top {args.top} overall on TUNE phone@25 --")
    for cfg in sorted(configs[2:], key=lambda c: -res[('tune', c)]['phone'])[:args.top]:
        print(row(cfg))
    print(f"  -- top {args.top} overall on TUNE f1@25 --")
    for cfg in sorted(configs[2:], key=lambda c: -res[('tune', c)]['f1'])[:args.top]:
        print(row(cfg))


if __name__ == "__main__":
    main()
