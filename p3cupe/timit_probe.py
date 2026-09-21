'''
In-training TIMIT boundary probe.

Scores real boundary placement every validation epoch, in seconds, instead of waiting hours for
the offline pipeline. The scoring function is IMPORTED from `modules/aligner/alignment_tests.py`
- `levenshtein_align` is the same monotonic one-to-one DP behind every published row in
the probe scorer, so each `val_timit_f1_25ms` value has a consistent definition.

TIMIT is validation-only. Training stays on whatever `data_src` says, so the comparison with
MFA (which is also trained without timestamps) stays fair.

## The four readouts, and what each one falsifies

| prefix   | what it reads                                          | what it decides |
|----------|--------------------------------------------------------|-----------------|
| (none)   | the Viterbi onsets, on the 5 ms frame grid             | the baseline number |
| `sub_`   | the same onsets, interpolated at the posterior crossing | **the headline** - did shaping the posteriors buy sub-frame precision? |
| `dip_`   | free peak-pick on the confidence dip, NO transcript     | did the V actually form? this is the model used as a plain segmenter |
| `flux_`  | free peak-pick on embedding cosine flux                 | an independent acoustic-change readout |
| `nobonus_` | the same aligner with the onset bonus OFF (holdout, frame grid) | **the model's own learning signal**: the bonus alone lifts a random-init model to ~70 f1@25, so the headline moves little when the model improves |
| `bonusflux_` | free peak-pick on the flux the onset bonus reads (`boundary_flux_source`) | with `tap`, falls if training erodes the front end; with `mel`, constant by construction |

If `sub_` does not beat `(none)`, the crossing is not where the boundary is and `ramp_frames`
is the knob. If `dip_` is far below `sub_`, the posteriors are not V-shaped and the score is
coming from the DP rather than from the model - the design has not done what it claims. If
`flux_` beats both, the trunk knows more about boundaries than its own posteriors express.

The onset bonus can make the decoder score well even when the model's own boundary signal is
weak. Read `nobonus_f1_25ms` alongside the headline to separate those effects.

**Read every readout against chance, not against zero.** All of them propose about as many
boundaries as there are, on a clip that has one every ~80 ms, so guessing scores well.
MEASURED on this exact architecture at random init, 400 clips:

    f1@25   decoder 22.54   sub 22.65   dip 29.86   flux 37.07
    abs_err decoder 27.86   sub 27.88

That is a reference floor. `flux` can start relatively high because spectral change does not
need training; `dip` and `sub` indicate whether the model expresses that information itself.

## Tune / holdout

The 400 clips are split by index parity: EVEN clips are the tuning half (logged as
`val_timit_tune_*`, the set to sweep decoder settings on), ODD clips are held out and are
what every headline `val_timit_*` key - and so `ckpt_monitor` - reports. Decoder settings
chosen on the same clips they are scored on would read optimistic. Numbers from before this
split are over all 400 clips and are not directly comparable.

The probe reads the JSON cache configured by `timit_probe_cache`.
'''

import json
import os
import sys

import lightning as L
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.levenshtein import levenshtein_align
from p3cupe.loss import (confidence_dip, embedding_delta, parabolic_refine, peak_pick)

SILENCE_CLASS = 0
DEFAULT_CACHE = "tmp/data/timit_probe_400.json"
DEFAULT_TOLERANCES = (10, 25, 50)


def load_cache(cache_path=DEFAULT_CACHE):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"TIMIT probe cache not found at {cache_path}. Build it once with:\n"
            f"    python utils/timit_probe_prep.py")
    with open(cache_path) as f:
        return json.load(f)


def _load_audio(path, sample_rate=16000):
    import torchaudio
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.squeeze(0)
    # Exactly `SegmentsCollator._rms_normalize`: divide by RMS, with no target scale.
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


def _onsets_ms(aligned, silence_class=SILENCE_CLASS):
    '''
    Predicted onsets in ms from `align_clip` output, dropping SIL onsets.

    SIL is not treated as a segmentation point.
    '''
    return sorted({int(round(ms)) for ph, ms in aligned
                   if ph != silence_class and ms > 0})


@torch.no_grad()
def score_model(pl_module, cache, batch_size=16, tolerances=DEFAULT_TOLERANCES,
                align_max_dist=100.0, max_clips=None):
    if max_clips is not None:
        cache = cache[:max_clips]

    device = next(pl_module.parameters()).device
    sample_rate = pl_module.sample_rate
    ms_per_frame = pl_module.ms_per_frame

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
            wavs = [_load_audio(c['audio'], sample_rate) for c in chunk]
            wav_lens = torch.tensor([w.numel() for w in wavs], dtype=torch.long)

            padded = torch.zeros(len(wavs), 1, int(wav_lens.max()))
            for i, w in enumerate(wavs):
                padded[i, 0, :w.numel()] = w
            padded = padded.to(device)

            spectral_lens = pl_module.calc_spec_len(wav_lens.to(device))
            logits_c, _, emb, bonus_flux = pl_module.encode_clip(
                padded, max_frames=int(spectral_lens.max()), return_embeddings=True,
                return_flux=True)
            log_probs = F.log_softmax(logits_c.float(), dim=2)
            spectral_lens = spectral_lens.clamp(max=log_probs.shape[1])

            seqs = [torch.tensor(c['ph'], dtype=torch.long) for c in chunk]
            seq_lens = torch.tensor([s.numel() for s in seqs], dtype=torch.long)
            padded_seq = torch.zeros(len(seqs), int(seq_lens.max()), dtype=torch.long)
            for i, s in enumerate(seqs):
                padded_seq[i, :s.numel()] = s

            # One DP run for the whole chunk, and both readouts come out of it: the integer
            # frame grid and the interpolated posterior crossing. Scoring them side by side is
            # what isolates the gain sub-frame precision is actually worth.
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
            ms = pl_module.ms_per_frame

            for i, entry in enumerate(chunk):
                sc = halves[(start + i) % 2 == 0]       # even index -> tune half
                dec, sub, dip, flux, bonusflux = (sc['dec'], sc['sub'], sc['dip'], sc['flux'],
                                                sc['bonusflux'])
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
                dec.add(known, pred, align_max_dist)
                if nobonus[i]:
                    nb = nobonus[i]
                    sc['nobonus'].add(known, _onsets_ms(
                        [(ph, raw * ms) for ph, raw, used
                         in zip(nb['phones'], nb['onsets'], nb['used']) if used]), align_max_dist)
                sub.add(known, _onsets_ms([(ph, ref * ms) for ph, _, ref in keep]),
                        align_max_dist)

                # --- transcript-free readouts, K matched to the aligner's own count ---------
                dip_signal = confidence_dip(log_probs[i, :n_frames])
                idx = peak_pick(dip_signal, k=len(pred), min_distance=2)
                dip.add(known, sorted({int(round((p - 0.5) * ms_per_frame))
                                       for p in parabolic_refine(dip_signal, idx)}),
                        align_max_dist)

                flux_signal = embedding_delta(emb[i, :n_frames])
                idx = peak_pick(flux_signal, k=len(pred), min_distance=2)
                flux.add(known, sorted({int(round((p - 0.5) * ms_per_frame))
                                        for p in parabolic_refine(flux_signal, idx)}),
                         align_max_dist)

                bonus_signal = bonus_flux[i, :n_frames]
                idx = peak_pick(bonus_signal, k=len(pred), min_distance=2)
                bonusflux.add(known, sorted({int(round((p - 0.5) * ms_per_frame))
                                           for p in parabolic_refine(bonus_signal, idx)}),
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

    def __init__(self, cache_path=DEFAULT_CACHE, batch_size=16, every_n_epochs=1,
                 tolerances=DEFAULT_TOLERANCES, max_clips=None, enabled=True):
        super().__init__()
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.tolerances = tuple(tolerances)
        self.max_clips = max_clips
        self.enabled = enabled
        self._cache = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.enabled or trainer.sanity_checking:
            return
        if (int(trainer.current_epoch) % self.every_n_epochs) != 0:
            return

        try:
            if self._cache is None:
                self._cache = load_cache(self.cache_path)
            results = score_model(pl_module, self._cache, batch_size=self.batch_size,
                                  tolerances=self.tolerances, max_clips=self.max_clips)
        except Exception as exc:                       # never let the probe kill a run
            if getattr(pl_module, 'main_node', True):
                print(f"WARN: TIMIT probe failed this epoch: {type(exc).__name__}: {exc}")
            return

        # Logged on EVERY rank, not rank-zero-only: `ckpt_monitor` is a val_timit_* key, and
        # under DDP a ModelCheckpoint on a rank where the metric is missing cannot monitor it.
        for key, value in results.items():
            pl_module.log(f'val_timit_{key}', float(value),
                          prog_bar=(key == f'sub_f1_{self.tolerances[1]}ms'),
                          on_step=False, on_epoch=True, sync_dist=True)

        if getattr(pl_module, 'main_node', True):
            tol = ", ".join(f"f1@{t}={results[f'sub_f1_{t}ms']:.2f}" for t in self.tolerances)
            print(f"[TIMIT probe] epoch {trainer.current_epoch}: sub-frame {tol}, "
                  f"abs_err={results['sub_abs_err']:.2f} ms, "
                  f"det={results['sub_detection_ratio']:.3f}, clips={int(results['clips'])}")
            print(f"[TIMIT probe]   frame-grid f1@25={results['f1_25ms']:.2f} "
                  f"(abs_err={results['abs_err']:.2f}) | "
                  f"dip f1@25={results['dip_f1_25ms']:.2f} "
                  f"(abs_err={results['dip_abs_err']:.2f}) | "
                  f"flux f1@25={results['flux_f1_25ms']:.2f} | "
                  f"bonusflux f1@25={results['bonusflux_f1_25ms']:.2f} | "
                  f"no-bonus f1@25={results['nobonus_f1_25ms']:.2f} | "
                  f"tune half sub f1@25={results['tune_sub_f1_25ms']:.2f}")
