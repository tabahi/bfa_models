'''
Clip-level forward and alignment readout, shared by training and inference.

`trainer.ComposedModel` (Lightning, training) and `inference.MbfaAligner` (no Lightning, the
published model) both call these functions, so the windowing, stitching, onset-bonus flux and
alignment readout a released model runs are the code it was trained and validated with - not a
second copy that can drift. Nothing here imports Lightning.

    ClipGeometry          the frame grid of one model (5 ms frames, 120 ms windows, 80 ms advance)
    encode_clip           windows -> MbfaCNN -> stitched per-frame logits (+ embeddings, flux)
    mel_flux              parameter-free log-mel change entering each frame (the onset bonus)
    decode_alignments     PeakLoss.align + posterior-crossing refinement, per clip
'''

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from p4mbfa import windowing
from p4mbfa.loss import refine_onsets


@dataclass(frozen=True)
class ClipGeometry:
    sample_rate: int
    window_size_ms: float
    stride_ms: float
    window_size_samples: int
    stride_samples: int
    frames_per_window: int
    stride_frames: int
    frame_stride_samples: int

    @property
    def ms_per_frame(self):
        return 1000.0 * self.frame_stride_samples / self.sample_rate

    @classmethod
    def for_model(cls, model, sample_rate, window_size_ms, stride_ms):
        '''Resolve the grid of `model` (an MbfaCNN); raises if it does not divide evenly.'''
        if not window_size_ms > stride_ms:
            raise ValueError(f"windows must overlap: window {window_size_ms} ms, "
                             f"stride {stride_ms} ms")
        stride_frames, stride_samples, window_samples = windowing.frames_geometry(
            window_size_ms=window_size_ms, stride_ms=stride_ms, sample_rate=sample_rate,
            frames_per_window=int(model.frames_per_window),
            frame_stride_samples=int(model.frame_stride_samples))
        if window_samples != int(model.input_wav_length):
            raise ValueError(f"window of {window_samples} samples, but the model was built for "
                             f"{model.input_wav_length}")
        return cls(int(sample_rate), float(window_size_ms), float(stride_ms), window_samples,
                   stride_samples, int(model.frames_per_window), stride_frames,
                   int(model.frame_stride_samples))

    def spectral_lengths(self, wav_lens):
        '''Valid stitched frames per clip, from the real (unpadded) sample lengths.'''
        return windowing.spectral_lengths(wav_lens, self.window_size_samples,
                                          self.stride_samples, self.frames_per_window,
                                          self.stride_frames)


def encode_clip(net, geom, audio_batch, flux_source='mel', max_frames=None,
                fixed_window_batch=False, return_embeddings=False, return_flux=False,
                cache=None):
    '''
    Windowed forward over a whole clip.

    Args:
        net: an MbfaCNN
        geom: its ClipGeometry
        audio_batch: [B, 1, L] padded waveforms
        flux_source: 'mel' (parameter-free log-mel) or 'tap' (the model's feature-extractor
            output), for the aligner's onset bonus
        max_frames: valid frames the caller actually needs, i.e. `spectral_lens.max()`; windows
            past it are not computed. IGNORED when `fixed_window_batch` is set, which keeps
            activation shapes (and GPU memory) constant - see p3cupe's notes.
        return_flux: adds [B, T_full], the acoustic change ENTERING each frame
        cache: a dict the caller keeps between calls, holding the mel transform per device
    Returns:
        (logits_ph, logits_phg, logits_tone, embeddings|None[, flux]), each [B, T_full, ...]
        where T_full is what the PADDED clip length implies.
    '''
    if flux_source not in ('mel', 'tap'):
        raise ValueError(f"flux_source must be 'mel' or 'tap', got {flux_source!r}")
    length = audio_batch.shape[-1]
    full_audio = audio_batch
    n_full = windowing.num_windows_for(length, geom.window_size_samples, geom.stride_samples)
    t_full = windowing.total_stitched_frames(n_full, geom.frames_per_window, geom.stride_frames)

    if max_frames is not None and not fixed_window_batch:
        # Every window that overlaps a needed frame, not merely enough to reach it.
        need = (max(int(max_frames), 1) - 1) // geom.stride_frames + 1
        n_win = min(n_full, max(1, need))
        audio_batch = audio_batch[..., :min(
            length, (n_win - 1) * geom.stride_samples + geom.window_size_samples)]

    windows = windowing.slice_windows(audio_batch, geom.sample_rate, geom.window_size_ms,
                                      geom.stride_ms)
    b, n_win, win_len = windows.shape

    if return_flux and flux_source == 'tap':
        emb, tap = net.encode(windows.reshape(-1, win_len), return_tap=True)
    else:
        emb = net.encode(windows.reshape(-1, win_len))
    logits = net.classify(emb)
    if logits[0].shape[1] != geom.frames_per_window:
        raise ValueError(f"expected {geom.frames_per_window} frames per window, "
                         f"got {logits[0].shape[1]}")

    def _stitch(x):
        if x is None:
            return None
        out = windowing.stitch_windows(
            x.reshape(b, n_win, geom.frames_per_window, -1),
            num_windows_total=n_win, frames_per_window=geom.frames_per_window,
            stride_frames=geom.stride_frames)
        pad = t_full - out.shape[1]
        return F.pad(out, (0, 0, 0, pad)) if pad > 0 else out

    out = tuple(_stitch(x) for x in logits) + (_stitch(emb if return_embeddings else None),)
    if not return_flux:
        return out
    if flux_source == 'mel':
        # From the UNTRIMMED audio: the max_frames shortcut cuts the tail after the last
        # window, and the mel frames near the clip end would then see a different edge padding
        # than training (which runs fixed_window_batch: true, i.e. untrimmed) - enough to move
        # the onset bonus, and so a boundary, by a frame.
        return out + (mel_flux(full_audio, t_full, geom, cache),)
    with torch.no_grad():
        e = F.normalize(tap.detach().float(), dim=-1)
        flux = torch.zeros(e.shape[:2], device=e.device)
        flux[:, 1:] = 1.0 - (e[:, 1:] * e[:, :-1]).sum(-1)
        valid = torch.ones_like(flux)
        valid[:, 0] = 0.0          # a window's first frame has no in-window predecessor
        st = _stitch(torch.stack([flux * valid, valid], dim=-1))
        flux = st[..., 0] / st[..., 1].clamp(min=1e-6)
        flux[:, 0] = 0.0
    return out + (flux,)


@torch.no_grad()
def mel_flux(audio_batch, t_full, geom, cache=None):
    '''
    L2 change of the log-mel spectrum ENTERING each frame, on the model's own frame grid:
    25 ms window, hop = one frame, `center=True`, so mel frame t is centred on sample t * 80
    exactly like conv frame t. [B, t_full], entry 0 = 0. No trainable parameters, so training
    cannot erode it.
    '''
    import torchaudio
    wav = audio_batch.reshape(audio_batch.shape[0], -1).float()
    cache = {} if cache is None else cache
    transform = cache.get(wav.device)
    if transform is None:
        transform = cache[wav.device] = torchaudio.transforms.MelSpectrogram(
            geom.sample_rate, n_fft=512, win_length=400, hop_length=geom.frame_stride_samples,
            n_mels=80, center=True).to(wav.device)
    with torch.autocast(device_type=wav.device.type, enabled=False):
        lm = (transform(wav) + 1e-6).log().transpose(1, 2)          # [B, frames, 80]
    n = min(lm.shape[1], t_full)
    flux = torch.zeros(wav.shape[0], t_full, device=wav.device)
    flux[:, 1:n] = (lm[:, 1:n] - lm[:, :n - 1]).norm(dim=-1)
    return flux


@torch.no_grad()
def decode_alignments(loss_fn, log_probs, seqs, seq_lens, spectral_lens, flux=None):
    '''
    Force-align a whole batch with the training-time decoder (one DP run for the batch).

    Args:
        loss_fn: the PeakLoss whose decoder settings to use
        log_probs: [B, T, C] ph-head log posteriors
        seqs: [B, S] local token ids, seq_lens: [B]
        spectral_lens: [B] valid frames per clip
        flux: [B, T] from `encode_clip(..., return_flux=True)`; required when the decoder's
            `bonus_weight` is set
    Returns:
        list of B dicts with `phones` (the extended sequence, SIL pads included), `onsets`
        (integer frame per phone), `refined` (float frame, interpolated at the posterior
        crossing) and `used` (False for a phone the path skipped). An empty dict marks a clip
        too short for its own sequence.
    '''
    al = loss_fn.align(log_probs, seqs, None, None, seq_lens, spectral_lens, flux=flux)
    ext, ext_lens, feasible = al['ext'], al['ext_lens'], al['feasible']
    onsets = al['onsets'].cpu()

    lp_cpu = log_probs.detach().float().cpu()
    ext_cpu, ext_lens_cpu = ext.cpu(), ext_lens.cpu()
    out = []
    for b in range(log_probs.shape[0]):
        n_ext, n_frames = int(ext_lens_cpu[b]), int(spectral_lens[b])
        if not bool(feasible[b]) or n_ext == 0:
            out.append({})
            continue
        refined = refine_onsets(lp_cpu[b, :n_frames], ext_cpu[b, :n_ext],
                                onsets[b, :n_ext + 1])
        out.append({
            'phones': ext_cpu[b, :n_ext].tolist(),
            'onsets': onsets[b, :n_ext].tolist(),
            'refined': [float(v) for v in refined],
            'used': [int(onsets[b, i + 1]) != int(onsets[b, i]) for i in range(n_ext)],
        })
    return out
