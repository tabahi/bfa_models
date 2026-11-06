'''
Inference for released p4mbfa models, without Lightning or the training yaml.

    from p4mbfa.inference import MbfaAligner

    aligner = MbfaAligner.from_pretrained(path)          # see `from_pretrained` for `path`
    wav = aligner.load_audio("clip.wav")
    out = aligner.encode(wav)                            # per-frame log posteriors, 3 heads
    segs = aligner.align(wav, ["SIL", "h", "ɛ", "l", "o"])   # forced alignment, ms

The clip forward and the alignment readout are `p4mbfa.clip` - the functions training and the
validation probe call - and the decoder is `loss.PeakLoss` built from the decoder settings the
model was trained with (config.json "decoder"), so a released model aligns exactly as it was
validated. Audio is RMS-normalised per clip, as `SegmentsCollator` does for training segments.

`class_prior_norm` (on for every released model) normalises the posteriors by their mean over the
BATCH being aligned, so the same clip can align slightly differently alone and in a batch; this is
also true of training and the probe.
'''

import json
import os

import torch
import torch.nn.functional as F

from p4mbfa import clip
from p4mbfa.export_hf import build_model, build_release_config, model_weights
from p4mbfa.loss import PeakLoss, SIL_CLASS


class MbfaAligner(torch.nn.Module):

    def __init__(self, config, state, device='cpu'):
        super().__init__()
        self.config = config
        self.model = build_model(config, state).eval()
        geo = config['geometry']
        self.sample_rate = int(geo['sample_rate'])
        self.geometry = clip.ClipGeometry.for_model(self.model, self.sample_rate,
                                                    float(geo['window_size_ms']),
                                                    float(geo['window_stride_ms']))
        dec = config['decoder']
        self.flux_source = str(dec['boundary_flux_source'])
        self.decoder = PeakLoss(
            min_frames=int(dec['min_phone_frames']), optional_sil=bool(dec['optional_sil']),
            class_prior_norm=bool(dec['class_prior_norm']),
            soft_frames=int(dec['soft_phone_frames']), soft_cost=float(dec['soft_frame_cost']),
            sil_skippable=bool(dec['sil_skippable']),
            bonus_weight=float(dec['boundary_bonus_weight']),
            contrast_frames=int(dec['boundary_contrast_frames']))
        labels = config['labels']
        self.lang_group = labels['lang_group']
        self.tokens = list(labels['tokens'])
        self.token_index = {t: i for i, t in enumerate(self.tokens)}
        self.phg_names = list(labels['phg_names'])
        self.tone_vocab = list(labels['tone_vocab'])
        self._mel_cache = {}
        self.to(device)

    @classmethod
    def from_pretrained(cls, path, device='cpu', revision=None):
        '''
        `path` is one of:
          - a directory written by `export_hf.py` (config.json + model.safetensors);
          - a p4mbfa training checkpoint (.ckpt / .pt) - needs standard_g2p for the token strings;
          - a Hugging Face repo id (e.g. "Tabahi/mbfa-latin"), downloaded with huggingface_hub.
            Only config.json and model.safetensors are fetched, not the repo's training
            checkpoint (ckpt/, for fine-tuning). `revision`: a branch, tag or commit.
        '''
        if os.path.isfile(path):
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            return cls(build_release_config(ckpt, path), model_weights(ckpt, path), device)
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path, revision=revision,
                                     allow_patterns=['config.json', 'model.safetensors'])
        from safetensors.torch import load_file
        with open(os.path.join(path, 'config.json')) as f:
            config = json.load(f)
        return cls(config, load_file(os.path.join(path, 'model.safetensors')), device)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def ms_per_frame(self):
        return self.geometry.ms_per_frame

    def load_audio(self, path):
        '''Mono float waveform at the model's sample rate, [L].'''
        import torchaudio
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav

    def token_ids(self, tokens):
        '''Token strings (or ids) -> ids. Unknown strings raise; nothing is mapped to <unk>.'''
        ids = []
        for t in tokens:
            if isinstance(t, int):
                if not 0 < t < len(self.tokens):
                    raise ValueError(f"token id {t} outside [1, {len(self.tokens)})")
                ids.append(t)
            elif t in self.token_index and self.token_index[t] != 0:
                ids.append(self.token_index[t])
            else:
                raise KeyError(f"{t!r} is not a {self.lang_group} token (or is <blank>)")
        return ids

    def _batch(self, audio, wav_lens, normalize):
        audio = torch.as_tensor(audio, dtype=torch.float32)
        if audio.dim() == 1:
            audio = audio[None]
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        if audio.dim() != 2:
            raise ValueError(f"audio must be [L], [B, L] or [B, 1, L], got {tuple(audio.shape)}")
        lens = (torch.full((audio.shape[0],), audio.shape[1], dtype=torch.long)
                if wav_lens is None else torch.as_tensor(wav_lens, dtype=torch.long))
        audio = audio.clone()
        for i, n in enumerate(lens.tolist()):
            audio[i, n:] = 0.0
            if normalize:     # exactly SegmentsCollator._rms_normalize, on the valid samples
                rms = audio[i, :n].pow(2).mean().sqrt()
                if rms > 0:
                    audio[i, :n] = audio[i, :n] / rms
        return audio[:, None].to(self.device), lens.to(self.device)

    @torch.no_grad()
    def encode(self, audio, wav_lens=None, normalize=True):
        '''
        Args:
            audio: [L], [B, L] or [B, 1, L] float at `sample_rate`; padded rows need `wav_lens`
            wav_lens: [B] valid samples per row (default: the full width)
        Returns:
            dict of [B, T, C] log posteriors 'ph', 'phg', 'tone', 'frames' [B] valid frames per
            clip, 'flux' [B, T] (the onset-bonus input) and 'ms_per_frame'
        '''
        audio, lens = self._batch(audio, wav_lens, normalize)
        frames = self.geometry.spectral_lengths(lens)
        ph, phg, tone, _, flux = clip.encode_clip(
            self.model, self.geometry, audio, flux_source=self.flux_source,
            max_frames=int(frames.max()), return_flux=True, cache=self._mel_cache)
        lp = lambda x: F.log_softmax(x.float(), dim=2)                 # noqa: E731
        return {'ph': lp(ph), 'phg': lp(phg), 'tone': lp(tone),
                'frames': frames.clamp(max=ph.shape[1]), 'flux': flux,
                'ms_per_frame': self.ms_per_frame}

    @torch.no_grad()
    def align(self, audio, tokens, sub_frame=True, normalize=True):
        '''
        Force-align one clip to its token sequence.

        Args:
            audio: [L] float at `sample_rate`
            tokens: local token strings or ids, in order (no <blank>); a leading/trailing SIL
                is added and skipped automatically when the audio has none
        Returns:
            list of {'token', 'token_id', 'start_ms', 'end_ms', 'is_sil'} for every phone the
            path visited, SIL included (an added SIL the path skipped is not listed). Raises if
            the clip is too short for its sequence.
        '''
        ids = self.token_ids(tokens)
        enc = self.encode(audio, normalize=normalize)
        n_frames = int(enc['frames'][0])
        seq = torch.tensor([ids], device=self.device)
        d = clip.decode_alignments(self.decoder, enc['ph'][:, :n_frames], seq,
                                   torch.tensor([len(ids)], device=self.device),
                                   torch.tensor([n_frames], device=self.device),
                                   flux=enc['flux'][:, :n_frames])[0]
        if not d:
            raise ValueError(f"{len(ids)} tokens do not fit in {n_frames} frames at "
                             f"min_phone_frames {self.decoder.min_frames}")
        pos = d['refined'] if sub_frame else [float(o) for o in d['onsets']]
        visited = [(ph, p) for ph, p, used in zip(d['phones'], pos, d['used']) if used]
        end_ms = n_frames * self.ms_per_frame
        out = []
        for i, (ph, p) in enumerate(visited):
            nxt = visited[i + 1][1] * self.ms_per_frame if i + 1 < len(visited) else end_ms
            out.append({'token': self.tokens[ph], 'token_id': int(ph),
                        'start_ms': float(p) * self.ms_per_frame, 'end_ms': float(nxt),
                        'is_sil': int(ph) == SIL_CLASS})
        return out
