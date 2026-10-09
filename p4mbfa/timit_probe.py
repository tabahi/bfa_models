'''
In-training TIMIT boundary probe (p3cupe's, on p4mbfa's label spaces).

Scores real boundary placement every validation epoch. TIMIT is validation-only. The cache is
built once by `p4mbfa/timit_probe_prep.py`: the p3cupe probe's 400 clips and ground-truth
boundaries, with the phone sequence re-phonemized into the language group's local tokens, since
forced alignment needs the sequence in the model's own dictionary (the p3cupe cache's ph66 ids
would be meaningless to a p4mbfa head). The probe raises if the cache was built for a different
language group or token count than the model's `ph_head`.

## The readouts

| prefix       | what it reads                                            |
|--------------|----------------------------------------------------------|
| (none)       | the Viterbi onsets, on the 5 ms frame grid                |
| `sub_`       | the same onsets, interpolated at the posterior crossing   |
| `dip_`       | free peak-pick on the ph-head confidence dip, NO transcript |
| `flux_`      | free peak-pick on embedding cosine flux                   |
| `nobonus_`   | the aligner with the onset bonus OFF: the model's own signal |
| `bonusflux_` | free peak-pick on the flux the onset bonus reads          |

Read every readout against chance, not zero (p3cupe, random init, 400 clips: f1@25 decoder 22.5,
dip 29.9, flux 37.1).

## Tune / holdout

EVEN clip indices are the tuning half (`val_timit_tune_*`, which `ckpt_monitor` selects on),
ODD clips are held out (every other `val_timit_*` key).
'''

import json
import os
import sys

import lightning as L
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.levenshtein import levenshtein_align                              # noqa: E402
from p4mbfa.loss import (SIL_CLASS, confidence_dip, embedding_delta,         # noqa: E402
                         parabolic_refine, peak_pick)

DEFAULT_TOLERANCES = (10, 25, 50)


def load_cache(cache_path, lang_group, n_tokens):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"TIMIT probe cache not found at {cache_path}. Build it once with:\n"
            f"    python p4mbfa/timit_probe_prep.py {lang_group}")
    with open(cache_path) as f:
        cache = json.load(f)
    if not isinstance(cache, dict) or 'clips' not in cache:
        raise ValueError(f"{cache_path} is not a p4mbfa probe cache (a p3cupe ph66 cache?); "
                         f"build one with p4mbfa/timit_probe_prep.py")
    if cache['lang_group'] != lang_group or int(cache['n_tokens']) != int(n_tokens):
        raise ValueError(f"{cache_path} was built for {cache['lang_group']} / "
                         f"{cache['n_tokens']} tokens, the model's ph_head is {lang_group} / "
                         f"{n_tokens}")
    return cache['clips']


def _load_audio(path, sample_rate=16000, start=None, end=None):
    '''
    One probe clip, RMS-normalised. `start` / `end` (seconds) cut a segment out of a longer
    file: a TIMIT clip is its whole file, but a Buckeye entry is a few seconds of a ~7 minute
    interview, and its `boundaries_ms` are relative to `start`. The cut comes BEFORE the
    normalisation, so the level is the segment's own, as it is for a training segment.
    '''
    import torchaudio
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.squeeze(0)
    if start is not None or end is not None:
        lo = int(round(float(start or 0.0) * sample_rate))
        hi = wav.numel() if end is None else int(round(float(end) * sample_rate))
        wav = wav[max(lo, 0):max(hi, 0)]
    # Exactly `SegmentsCollator._rms_normalize`.
    rms = wav.pow(2).mean().sqrt()
    if rms > 0:
        wav = wav / rms
    return wav


class _Scorer:
    '''Accumulates one readout's onset scores over the epoch.'''

    def __init__(self, tolerances):
        self.tolerances = tolerances
        self.known = 0
        self.pred = 0
        self.clips = 0
        self.hits = {t: 0 for t in tolerances}
        self.abs_errors = []

    def add(self, known, pred, max_dist=100.0):
        if not pred or not known:
            return
        self.known += len(known)
        self.pred += len(pred)
        self.clips += 1
        for _, _, signed in levenshtein_align(known, pred, max_dist=max_dist):
            err = abs(signed)
            self.abs_errors.append(err)
            for t in self.tolerances:
                if err <= t:
                    self.hits[t] += 1

    def results(self, prefix='', brief=False):
        out = {f'{prefix}abs_err': (sum(self.abs_errors) / len(self.abs_errors))
               if self.abs_errors else 0.0}
        if not brief:
            out.update({
                f'{prefix}clips': float(self.clips),
                f'{prefix}known': float(self.known),
                f'{prefix}pred': float(self.pred),
                f'{prefix}detection_ratio': (self.pred / self.known) if self.known else 0.0,
            })
        for t in self.tolerances:
            recall = (self.hits[t] / self.known * 100) if self.known else 0.0
            precision = (self.hits[t] / self.pred * 100) if self.pred else 0.0
            f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0
            out[f'{prefix}f1_{t}ms'] = f1
            if not brief:
                out[f'{prefix}recall_{t}ms'] = recall
                out[f'{prefix}precision_{t}ms'] = precision
        return out


def _onsets_ms(aligned):
    '''Predicted onsets in ms from `align_clip`-style output, dropping SIL onsets.'''
    return sorted({int(round(ms)) for ph, ms in aligned if ph != SIL_CLASS and ms > 0})


@torch.no_grad()
def score_model(pl_module, cache, batch_size=16, tolerances=DEFAULT_TOLERANCES,
                align_max_dist=100.0, max_clips=None, onset_trace=None):
    '''
    `onset_trace`: an optional dict, filled with `{clip index: [refined onset frames]}`, for
    measuring how far the segmentation moved between epochs.
    '''
    if max_clips is not None:
        cache = cache[:max_clips]

    device = next(pl_module.parameters()).device
    sample_rate = pl_module.sample_rate
    ms = pl_module.ms_per_frame

    was_training = pl_module.training
    pl_module.eval()

    # [holdout, tune]: index parity, see the module docstring
    halves = [{name: _Scorer(tolerances)
               for name in ('dec', 'sub', 'dip', 'flux', 'bonusflux', 'nobonus')}
              for _ in range(2)]
    need_flux = bool(pl_module.loss_fn.bonus_weight)

    try:
        for start in range(0, len(cache), batch_size):
            chunk = cache[start:start + batch_size]
            wavs = [_load_audio(c['audio'], sample_rate, c.get('start'), c.get('end'))
                    for c in chunk]
            wav_lens = torch.tensor([w.numel() for w in wavs], dtype=torch.long)

            padded = torch.zeros(len(wavs), 1, int(wav_lens.max()))
            for i, w in enumerate(wavs):
                padded[i, 0, :w.numel()] = w
            padded = padded.to(device)

            spectral_lens = pl_module.calc_spec_len(wav_lens.to(device))
            logits_ph, _, _, emb, bonus_flux = pl_module.encode_clip(
                padded, max_frames=int(spectral_lens.max()), return_embeddings=True,
                return_flux=True)
            log_probs = F.log_softmax(logits_ph.float(), dim=2)
            spectral_lens = spectral_lens.clamp(max=log_probs.shape[1])

            seqs = [torch.tensor(c['local_ph'], dtype=torch.long) for c in chunk]
            seq_lens = torch.tensor([s.numel() for s in seqs], dtype=torch.long)
            padded_seq = torch.zeros(len(seqs), int(seq_lens.max()), dtype=torch.long)
            for i, s in enumerate(seqs):
                padded_seq[i, :s.numel()] = s

            details = pl_module.align_batch(log_probs, padded_seq.to(device),
                                            seq_lens.to(device), spectral_lens,
                                            flux=bonus_flux if need_flux else None)
            nobonus = details
            if need_flux:
                saved_w, pl_module.loss_fn.bonus_weight = pl_module.loss_fn.bonus_weight, 0.0
                try:
                    nobonus = pl_module.align_batch(log_probs, padded_seq.to(device),
                                                    seq_lens.to(device), spectral_lens)
                finally:
                    pl_module.loss_fn.bonus_weight = saved_w

            for i, entry in enumerate(chunk):
                sc = halves[(start + i) % 2 == 0]       # even index -> tune half
                n_frames = int(spectral_lens[i])
                known = sorted(set(entry['boundaries_ms']))
                detail = details[i]
                if not known or not detail:
                    continue
                keep = [(ph, raw, ref) for ph, raw, ref, used
                        in zip(detail['phones'], detail['onsets'], detail['refined'],
                               detail['used']) if used]
                pred = _onsets_ms([(ph, raw * ms) for ph, raw, _ in keep])
                if not pred:
                    continue
                if onset_trace is not None:
                    onset_trace[start + i] = [float(ref) for _, _, ref in keep]
                sc['dec'].add(known, pred, align_max_dist)
                if nobonus[i]:
                    nb = nobonus[i]
                    sc['nobonus'].add(known, _onsets_ms(
                        [(ph, raw * ms) for ph, raw, used
                         in zip(nb['phones'], nb['onsets'], nb['used']) if used]), align_max_dist)
                sc['sub'].add(known, _onsets_ms([(ph, ref * ms) for ph, _, ref in keep]),
                              align_max_dist)

                # --- transcript-free readouts, K matched to the aligner's own count ---------
                for name, signal in (('dip', confidence_dip(log_probs[i, :n_frames])),
                                     ('flux', embedding_delta(emb[i, :n_frames])),
                                     ('bonusflux', bonus_flux[i, :n_frames])):
                    idx = peak_pick(signal, k=len(pred), min_distance=2)
                    sc[name].add(known, sorted({int(round((p - 0.5) * ms))
                                                for p in parabolic_refine(signal, idx)}),
                                 align_max_dist)
    finally:
        if was_training:
            pl_module.train()

    held, tune = halves
    results = held['dec'].results()
    results.update(held['sub'].results('sub_'))
    results.update(held['dip'].results('dip_', brief=True))
    results.update(held['flux'].results('flux_', brief=True))
    results.update(held['bonusflux'].results('bonusflux_', brief=True))
    results.update(held['nobonus'].results('nobonus_', brief=True))
    results.update(tune['dec'].results('tune_', brief=True))
    results.update(tune['sub'].results('tune_sub_', brief=True))
    return results


class TimitAlignProbe(L.Callback):
    '''Scores TIMIT boundary placement at the end of every validation epoch.'''

    def __init__(self, cache_path, lang_group, n_tokens, batch_size=16, every_n_epochs=1,
                 tolerances=DEFAULT_TOLERANCES, max_clips=None):
        super().__init__()
        self.batch_size = batch_size
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.tolerances = tuple(tolerances)
        self.max_clips = max_clips
        # Loaded and checked at construction, so a missing or mismatched cache fails before
        # training starts rather than after the first epoch.
        self._cache = load_cache(cache_path, lang_group, n_tokens)
        self._prev_onsets = None

    def _drift(self, trace):
        '''How far this epoch's segmentation moved from last epoch's, on the same clips.'''
        prev = self._prev_onsets
        if not prev or not trace:
            return {}
        deltas = [abs(a - b) for k, cur in trace.items()
                  for a, b in zip(cur, prev.get(k, ())) if len(prev.get(k, ())) == len(cur)]
        if not deltas:
            return {}
        deltas.sort()
        return {'onset_drift_frames': sum(deltas) / len(deltas),
                'onset_drift_p95_frames': deltas[min(len(deltas) - 1, int(0.95 * len(deltas)))]}

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if (int(trainer.current_epoch) % self.every_n_epochs) != 0:
            return

        trace = {}
        results = score_model(pl_module, self._cache, batch_size=self.batch_size,
                              tolerances=self.tolerances, max_clips=self.max_clips,
                              onset_trace=trace)
        results.update(self._drift(trace))
        self._prev_onsets = trace

        # Logged on EVERY rank: `ckpt_monitor` is a val_timit_* key, and under DDP a
        # ModelCheckpoint on a rank where the metric is missing cannot monitor it.
        for key, value in results.items():
            pl_module.log(f'val_timit_{key}', float(value),
                          prog_bar=(key == f'sub_f1_{self.tolerances[1]}ms'),
                          on_step=False, on_epoch=True, sync_dist=True)

        if getattr(pl_module, 'main_node', True):
            tol = ", ".join(f"f1@{t}={results[f'sub_f1_{t}ms']:.2f}" for t in self.tolerances)
            print(f"[TIMIT probe] epoch {trainer.current_epoch}: sub-frame {tol}, "
                  f"abs_err={results['sub_abs_err']:.2f} ms, "
                  f"det={results['sub_detection_ratio']:.3f}, clips={int(results['clips'])}")
            print(f"[TIMIT probe]   frame-grid f1@25={results['f1_25ms']:.2f} | "
                  f"dip f1@25={results['dip_f1_25ms']:.2f} | "
                  f"flux f1@25={results['flux_f1_25ms']:.2f} | "
                  f"bonusflux f1@25={results['bonusflux_f1_25ms']:.2f} | "
                  f"no-bonus f1@25={results['nobonus_f1_25ms']:.2f} | "
                  f"tune f1@25={results['tune_f1_25ms']:.2f}")
            if 'onset_drift_frames' in results:
                print(f"[TIMIT probe]   alignment moved "
                      f"{results['onset_drift_frames']:.2f} frames since last epoch "
                      f"(p95 {results['onset_drift_p95_frames']:.2f})")
