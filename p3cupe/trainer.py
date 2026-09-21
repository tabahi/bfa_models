'''
Lightning module for p3cupe.

Its whole job: slice a clip into context-free windows, run the CNN on every window, stitch the
per-window outputs back onto one 5 ms frame grid, and hand the result to `PeakLoss`. Externally
it behaves as if the model had processed the whole clip; internally nothing ever saw more than
120 ms, which is what keeps the model from learning any single language's phonotactics.

The trainer focuses on one objective: the segmental phoneme loss, with no blank penalty,
partition loss, flux hinge, boundary head, or secondary priority controller. There is one loss,
one group-loss weight, and two explicit schedules: the optimizer and alignment prior.
'''

import time

import lightning as L
import torch
import torch.nn.functional as F
from argparse import Namespace
from torch import optim

from p3cupe import windowing
from p3cupe.loss import PeakLoss, refine_onsets
from p3cupe.metrics import MetricsCalculator5
from p3cupe.model import CupeCNN


class ComposedModel(L.LightningModule):

    def __init__(self, hp=None):
        super().__init__()
        self.model_ready = False
        self.main_node = True
        self.last_epoch_time = time.time()
        self.metrics_cum = {}
        self.has_boundaries = False

        if hp is not None:
            if isinstance(hp, dict):
                hp = Namespace(**hp)
            self.hp = hp
            self.save_hyperparameters(self.hp)
            self.build_model()

    # ------------------------------------------------------------------ build

    def build_model(self):
        print("............................. building model")
        self.sample_rate = int(getattr(self.hp, 'sample_rate', 16000))
        self.window_size_ms = float(getattr(self.hp, 'window_size_ms', 120))
        self.stride_ms = float(getattr(self.hp, 'window_stride_ms', 80))
        assert self.window_size_ms > self.stride_ms, "windows must overlap"

        self.window_size_wav = int(round(self.window_size_ms * self.sample_rate / 1000))
        self.stride_size_wav = int(round(self.stride_ms * self.sample_rate / 1000))

        # Constant activation shapes every step, so GPU memory does not depend on which clips
        # land in a batch. See `encode_clip` for the measurement behind this.
        self.fixed_window_batch = bool(getattr(self.hp, 'fixed_window_batch', True))

        self.model = CupeCNN(
            input_wav_length=self.window_size_wav,
            cnn_channels=int(getattr(self.hp, 'cnn_channels', 192)),
            embed_dim=int(getattr(self.hp, 'embed_dim', 512)),
            context_kernels=list(getattr(self.hp, 'context_kernels', [5, 3])),
            cnn_dropout=float(getattr(self.hp, 'cnn_dropout', 0.1)),
            head_dropout=float(getattr(self.hp, 'head_dropout', 0.1)),
            noise_level=float(getattr(self.hp, 'noise_level', 0.01)),
            phoneme_classes=int(self.hp.phoneme_classes),
            phoneme_groups=int(self.hp.phoneme_groups),
            # 0 = the original 5.1M trunk; a yaml without these keys builds that.
            frame_blocks=int(getattr(self.hp, 'frame_blocks', 0)),
            ffn_mult=int(getattr(self.hp, 'ffn_mult', 4)),
            attn_radius=int(getattr(self.hp, 'attn_radius', 0)),
            attn_heads=int(getattr(self.hp, 'attn_heads', 8)),
            block_dropout=float(getattr(self.hp, 'block_dropout', 0.1)),
        )

        self._resolve_geometry()
        self.loss_fn = PeakLoss(
            min_frames=int(getattr(self.hp, 'min_phone_frames', 2)),
            ramp_frames=float(getattr(self.hp, 'ramp_frames', 3.0)),
            group_weight=float(getattr(self.hp, 'group_loss_weight', 0.3)),
            optional_sil=bool(getattr(self.hp, 'optional_sil', True)),
            prior_sigma=float(getattr(self.hp, 'align_prior_sigma', 0.1)),
            prior_strength=float(getattr(self.hp, 'align_prior_strength', 1.0)),
            class_prior_norm=bool(getattr(self.hp, 'class_prior_norm', False)),
        )
        self._read_decoder_hp()

        self.params_count = sum(p.numel() for p in self.parameters())
        self.exp_name = f"{self.hp.experiment}_{self.hp.data_src}_{self.window_size_ms:g}ms"
        print(self.model.describe(self.sample_rate))
        print("............................. model built.")
        self.model_ready = True
        return True

    def _read_decoder_hp(self):
        '''
        The segmental decoder's knobs (see `loss.viterbi_align`), read from the LIVE hp both at
        build and in `configure()`. Every one's default is the plain min-duration chain.
        '''
        self.loss_fn.soft_frames = int(getattr(self.hp, 'soft_phone_frames', 0))
        self.loss_fn.soft_cost = float(getattr(self.hp, 'soft_frame_cost', 1.0))
        self.loss_fn.sil_skippable = bool(getattr(self.hp, 'sil_skippable', False))
        self.loss_fn.bonus_weight = float(getattr(self.hp, 'boundary_bonus_weight', 0.0))
        self.loss_fn.contrast_frames = int(getattr(self.hp, 'boundary_contrast_frames', 17))
        # Soft targets: 0 = hard-EM on the Viterbi path alone (see loss.alignment_occupancy).
        self.loss_fn.soft_target_mix = float(getattr(self.hp, 'soft_target_mix', 0.0))
        self.loss_fn.soft_target_scale = float(getattr(self.hp, 'soft_target_scale', 0.3))

    def _resolve_geometry(self):
        '''
        Pin down the frame grid, and refuse to run on one that does not divide evenly.

        `frames_geometry` raises rather than warns because a geometry error would shift every
        reported timestamp.
        '''
        self.frames_per_window = int(self.model.frames_per_window)
        self.frame_stride_samples = int(self.model.frame_stride_samples)
        self.stride_frames, _, _ = windowing.frames_geometry(
            window_size_ms=self.window_size_ms,
            stride_ms=self.stride_ms,
            sample_rate=self.sample_rate,
            frames_per_window=self.frames_per_window,
            frame_stride_samples=self.frame_stride_samples,
        )
        self.ms_per_frame = 1000.0 * self.frame_stride_samples / self.sample_rate
        print(f"Geometry: {self.ms_per_frame:.4f} ms/frame, {self.frames_per_window} "
              f"frames/window, window advance {self.stride_frames} frames "
              f"({self.stride_ms} ms), receptive field "
              f"{1000.0 * self.model.receptive_field_samples / self.sample_rate:.1f} ms")

    def update_hyperparams(self, hp):
        '''Re-read the live yaml after a weights-only warm start.

        On a warm start Lightning calls `on_load_checkpoint`, which rebuilds the model from the
        CHECKPOINT's hyperparameters. The live YAML is restored explicitly before training.
        `configure()` re-reads the loss settings from `self.hp` after this for the same reason.
        '''
        self.hp = hp
        self.save_hyperparameters(self.hp)
        self.last_epoch_time = time.time()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['model_config'] = self.model.save_config_state_dict()
        return checkpoint

    ADDITIVE_ARCH_KEYS = {'attn_radius', 'attn_heads', 'frame_blocks'}

    def on_load_checkpoint(self, checkpoint):
        '''
        Architecture comes from the CHECKPOINT, not from the yaml.

        `build_model` below runs on the checkpoint's own hyperparameters, and then
        `model_config` - the shapes that were actually saved - overrides them where they
        disagree. Both facts are printed, because the alternative (a strict state_dict load
        raising a wall of shape errors, or worse, a quiet mismatch) is how this project
        previously spent months running configs nobody was reading. Loss settings ARE re-read
        from the live yaml, in `configure()`.

        ## warm_start_arch_overrides

        The one exception: a yaml list of ADDITIVE architecture keys the live yaml may change on
        a warm start (`ADDITIVE_ARCH_KEYS`). Each only adds `FrameBlock` parameters whose
        residual output is zero-initialised, so the grown model computes exactly what the
        checkpoint computed until training moves them. The added
        parameters are taken from the freshly built model, and anything missing from the
        checkpoint that is NOT such a parameter still fails the strict load.
        '''
        live_hp = getattr(self, 'hp', None)          # the yaml passed to load_from_checkpoint
        hp = checkpoint['hyper_parameters']
        self.hp = Namespace(**hp) if isinstance(hp, dict) else hp
        print("Loading from checkpoint")
        self.build_model()

        saved = checkpoint.get('model_config')
        overrides = {}
        for key in (getattr(live_hp, 'warm_start_arch_overrides', None) or []):
            if key not in self.ADDITIVE_ARCH_KEYS:
                raise ValueError(f"warm_start_arch_overrides: {key!r} is not additive; only "
                                 f"{sorted(self.ADDITIVE_ARCH_KEYS)} can change on a warm start")
            overrides[key] = getattr(live_hp, key)
        if saved and overrides:
            grown = {**saved, **overrides}
            if int(grown.get('frame_blocks', 0)) < int(saved.get('frame_blocks', 0)):
                raise ValueError("warm_start_arch_overrides cannot remove frame_blocks")
            print("Warm start GROWS the checkpoint's architecture (zero-initialised additions):")
            for k in overrides:
                print(f"   {k}: checkpoint={saved.get(k, 0)} -> yaml={overrides[k]}")
            saved = grown

        if saved:
            differing = {k: (self.model.config.get(k), v) for k, v in saved.items()
                         if self.model.config.get(k) != v}
            if differing:
                print("Checkpoint architecture differs from its hp; building the saved"
                      + (" + grown" if overrides else "") + " shapes:")
                for k, (mine, theirs) in differing.items():
                    print(f"   {k}: {mine} -> {theirs}")
                self.model.load_config_state_dict(saved)
                self.model.make_model()
                self._resolve_geometry()

        if 'state_dict' in checkpoint:
            checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items()
                                        if not k.startswith('loss_fn')}
            if overrides:
                old_blocks = int(checkpoint.get('model_config', {}).get('frame_blocks', 0))
                own = self.state_dict()
                added = [k for k in own if k not in checkpoint['state_dict']]

                def is_addition(k):
                    parts = k.split('.')
                    if parts[:2] != ['model', 'blocks']:
                        return False
                    return int(parts[2]) >= old_blocks or parts[3] in ('attn', 'attn_norm')

                stray = [k for k in added if not is_addition(k)]
                if stray:
                    raise ValueError(f"checkpoint is missing non-additive weights: {stray[:8]}")
                for k in added:
                    checkpoint['state_dict'][k] = own[k]
                print(f"   {len(added)} added tensors start from their zero-initialised values")

    def configure(self, ph_seq_max, train_wav_len_max, val_wav_len_max=None,
                  train_total=0, val_total=0, test_total=0, main_node=True):
        print("............................. configure")
        self.main_node = main_node
        self.wav_len_max = int(train_wav_len_max)

        # Re-read the loss settings from the LIVE hyperparameters (see update_hyperparams).
        self.loss_fn.min_frames = int(getattr(self.hp, 'min_phone_frames', 2))
        self.loss_fn.ramp_frames = float(getattr(self.hp, 'ramp_frames', 3.0))
        self.loss_fn.group_weight = float(getattr(self.hp, 'group_loss_weight', 0.3))
        self.loss_fn.prior_sigma = float(getattr(self.hp, 'align_prior_sigma', 0.1))
        self.loss_fn.prior_strength = float(getattr(self.hp, 'align_prior_strength', 1.0))
        self.loss_fn.class_prior_norm = bool(getattr(self.hp, 'class_prior_norm', False))
        self._read_decoder_hp()

        max_spec_len = int(self.calc_spec_len(torch.tensor([train_wav_len_max])).item())
        max_spec_len_val = int(self.calc_spec_len(
            torch.tensor([val_wav_len_max if val_wav_len_max else train_wav_len_max])).item())
        print(f"wav_len_max: {train_wav_len_max} samples "
              f"({train_wav_len_max / self.sample_rate:.1f} s) -> {max_spec_len} frames; "
              f"ph_seq_max: {ph_seq_max}")
        assert max_spec_len >= ph_seq_max * self.loss_fn.min_frames, (
            "clips are too short for their phoneme sequences at this min_phone_frames")

        # Fail here, at startup, rather than in the backward pass after the data pipeline has
        # spent ten minutes warming up.
        #self.check_batch_feasible(int(getattr(self.hp, 'batch_size', 1)), max_spec_len)

        # PER/GER are scored on at most `per_max_segments` clips PER RANK, which is a separate
        # and much smaller number than `val_max_segments` (that one sets the val_loss
        # population). See `per_capacity` below for why the two are not the same knob.
        per_cap = int(getattr(self.hp, 'per_max_segments', 0))
        self.per_capacity = min(val_total, per_cap) if per_cap > 0 else val_total
        if self.per_capacity < val_total and bool(getattr(self.hp, 'per_enabled', True)):
            print(f"PER/GER on the first {self.per_capacity} of {val_total} val clips per rank "
                  f"(val_loss still uses all of them)")

        light = bool(getattr(self.hp, 'light_val_metrics', True))
        self.metrics_cum = {}
        # PER/GER: greedy decoding with no blank measures argmax flicker and is slow.
        per_modes = (('val', self.per_capacity, max_spec_len_val),
                     ('test', test_total, max_spec_len))
        if not bool(getattr(self.hp, 'per_enabled', True)):
            print("PER/GER disabled (per_enabled: false); val_frame_acc is the identity guard")
            per_modes = ()
        for mode, total, spec in per_modes:
            self.metrics_cum[mode] = MetricsCalculator5(
                blank_class=self.hp.phoneme_classes, total_classes=self.hp.phoneme_classes + 1,
                blank_group=self.hp.phoneme_groups, total_groups=self.hp.phoneme_groups + 1,
                total_entries=max(int(total), 1), ph_seq_max=ph_seq_max, max_spec_len=spec,
                mode=getattr(self.hp, 'per_mode', 'greedy'), device='cpu',
                use_pred_lengths=True, light_metrics=light)
        print("------------------------------------------------------------------")
        return self.model

    RESERVED_PER_EST = 1.15      # reserved GB per GB of estimated activations (see below)
    RESERVED_FLOOR_GB = 0.5      # CUDA context, weights, optimiser state

    

    def per_epoch_active(self):
        '''
        Whether PER/GER are scored this epoch at all.

        They are GUARDS here, not the selection metric - `ckpt_monitor` is a boundary f1 and
        `early_stop_metric` is `val_loss`, so nothing depends on PER existing every epoch.
        '''
        n = int(getattr(self.hp, 'per_every_n_epochs', 1))
        return n <= 1 or (int(self.current_epoch) % n) == 0

    # ------------------------------------------------------------------ geometry helpers

    def calc_spec_len(self, wav_lens):
        '''Valid stitched frames per clip, from the real sample lengths.'''
        return windowing.spectral_lengths(
            wav_lens, self.window_size_wav, self.stride_size_wav,
            self.frames_per_window, self.stride_frames)

    def samples_to_frames(self, samples):
        '''
        Sample position -> stitched frame index.

        One line, because the stitched grid IS the conv grid: frame f is centred on sample
        f * frame_stride_samples. `0` marks padding in the boundary tensors and stays `0`.
        '''
        frames = torch.round(samples.float() / self.frame_stride_samples).long()
        return frames * (samples > 0).long()

    # ------------------------------------------------------------------ forward

    def encode_clip(self, audio_batch, max_frames=None, return_embeddings=False,
                    return_flux=False):
        '''
        Windowed forward over a whole clip.

        Args:
            audio_batch: [B, 1, L] padded waveforms
            max_frames: valid frames the caller actually needs, i.e. `spectral_lens.max()`.
                Windows past that are pure zero padding, so they are not computed at all and
                their frames come back as zeros. `SegmentsCollator` pads every clip to
                `duration_max` seconds, so on a word-level source (1 s clips, `duration_max:
                10`) this alone is a ~10x saving on the dominant cost in the whole model.
                None computes every window, which is what the padded shape implies.
                IGNORED when `fixed_window_batch` is set - see below.

        ## fixed_window_batch

        The shortcut makes the activation shapes depend on the batch's contents, and therefore
        makes GPU memory depend on which clips happen to land together. That is fine for
        throughput and bad for operating a card at 95%: you must leave headroom for the worst
        batch you have not met yet, and the footprint keeps moving. MEASURED over 30 steps with
        clip lengths drawn from 1-10 s, reserved memory after warmup:

            ragged shapes, default allocator      4.03 - 4.46 GB   (0.43 GB swing)
            ragged shapes, expandable_segments    3.10 - 3.14 GB   (0.04 GB swing)
            FIXED shapes,  default allocator      3.95 GB          (0.00 GB swing)
            FIXED shapes,  expandable_segments    3.23 GB          (0.00 GB swing)

        So `fixed_window_batch: true` buys a footprint that does not move at all, at the price
        of computing every clip at the padded width - which means `duration_max` must be set to
        what the data actually contains, or you pay for silence. That is the trade; it is in
        the yaml, not hard-coded.

        ## return_flux

        Adds a 4th output, [B, T_full]: cosine change of the `feature_extractor` tap ENTERING
        each frame, for the aligner's onset bonus. Computed inside each window (detached, a
        [N, 24] signal) and then stitched, rather than stitching the 512-d tap and differencing
        it: that tensor would be ~440 MB at a training batch, on a card sized to 94%. A
        window's first frame has no predecessor inside the window, so it is masked out of the
        stitch and that frame takes its value from the overlapping window instead.
        Returns:
            (logits_classes, logits_groups, embeddings|None[, flux]), each [B, T_full, ...]
            where T_full is what the PADDED clip length implies - so the shape does not vary
            with the batch and `MetricsCalculator5`'s fixed buffers still line up.
        '''
        length = audio_batch.shape[-1]
        n_full = windowing.num_windows_for(length, self.window_size_wav, self.stride_size_wav)
        t_full = windowing.total_stitched_frames(n_full, self.frames_per_window,
                                                 self.stride_frames)

        n_win = n_full
        if max_frames is not None and not self.fixed_window_batch:
            # Enough windows that every frame below `max_frames` gets ALL the windows that
            # overlap it, not merely enough to reach it: frame f is stitched from every window
            # w with w * stride_frames <= f, so the last one needed is floor((f_max) /
            # stride_frames). Stopping one short changes the value of the last ~8 frames of
            # every clip, which is a real change to the model's output, not a rounding.
            need = (max(int(max_frames), 1) - 1) // self.stride_frames + 1
            n_win = min(n_full, max(1, need))
            audio_batch = audio_batch[..., :min(
                length, (n_win - 1) * self.stride_size_wav + self.window_size_wav)]

        windows = windowing.slice_windows(audio_batch, self.sample_rate,
                                          self.window_size_ms, self.stride_ms)
        b, n_win, win_len = windows.shape

        flux_source = str(getattr(self.hp, 'boundary_flux_source', 'tap'))
        if return_flux and flux_source == 'tap':
            emb, tap = self.model.encode(windows.reshape(-1, win_len), return_tap=True)
        else:
            emb = self.model.encode(windows.reshape(-1, win_len))
        logits_c, logits_g = self.model.classify(emb)
        if logits_c.shape[1] != self.frames_per_window:
            raise ValueError(f"expected {self.frames_per_window} frames per window, "
                             f"got {logits_c.shape[1]}")

        def _stitch(x):
            if x is None:
                return None
            out = windowing.stitch_windows(
                x.reshape(b, n_win, self.frames_per_window, -1),
                num_windows_total=n_win, frames_per_window=self.frames_per_window,
                stride_frames=self.stride_frames)
            pad = t_full - out.shape[1]
            return F.pad(out, (0, 0, 0, pad)) if pad > 0 else out

        out = (_stitch(logits_c), _stitch(logits_g), _stitch(emb if return_embeddings else None))
        if not return_flux:
            return out
        if flux_source == 'mel':
            return out + (self._mel_flux(audio_batch, t_full),)
        if flux_source != 'tap':
            raise ValueError(f"boundary_flux_source must be 'mel' or 'tap', got {flux_source!r}")
        with torch.no_grad():
            e = F.normalize(tap.detach().float(), dim=-1)
            flux = torch.zeros(e.shape[:2], device=e.device)
            flux[:, 1:] = 1.0 - (e[:, 1:] * e[:, :-1]).sum(-1)
            valid = torch.ones_like(flux)
            valid[:, 0] = 0.0
            st = _stitch(torch.stack([flux * valid, valid], dim=-1))
            flux = st[..., 0] / st[..., 1].clamp(min=1e-6)
            flux[:, 0] = 0.0
        return out + (flux,)

    @torch.no_grad()
    def _mel_flux(self, audio_batch, t_full):
        '''
        L2 change of the log-mel spectrum ENTERING each frame, on the model's own frame grid:
        25 ms window, hop = one frame (80 samples), `center=True`, so mel frame t is centred on
        sample t * 80 exactly like conv frame t. [B, t_full], entry 0 = 0.

        Why this and not the model's own tap (`boundary_flux_source: tap`): it has no trainable
        parameters, so training cannot change its flux signal.
        '''
        import torchaudio
        wav = audio_batch.reshape(audio_batch.shape[0], -1).float()
        mel = getattr(self, '_mel_cache', None)
        if mel is None or mel[0] != wav.device:
            # Held outside the module tree on purpose: registered, its buffers would enter the
            # state_dict and every checkpoint from before this existed would stop loading.
            t = torchaudio.transforms.MelSpectrogram(
                self.sample_rate, n_fft=512, win_length=400, hop_length=self.frame_stride_samples,
                n_mels=80, center=True).to(wav.device)
            self._mel_cache = mel = (wav.device, t)
        with torch.autocast(device_type=wav.device.type, enabled=False):
            lm = (mel[1](wav) + 1e-6).log().transpose(1, 2)          # [B, frames, 80]
        n = min(lm.shape[1], t_full)
        flux = torch.zeros(wav.shape[0], t_full, device=wav.device)
        flux[:, 1:n] = (lm[:, 1:n] - lm[:, :n - 1]).norm(dim=-1)
        return flux

    def prior_weight_now(self):
        '''
        Diagonal-prior weight for the current epoch: 1 -> 0 linearly over
        `align_prior_epochs`, then off for good.

        This is the flat start. It is the ONLY schedule in the objective, and it exists because
        a hard-EM alignment read off a randomly initialised model is noise. Set
        `align_prior_epochs: 0` when warm starting from a checkpoint that already aligns.
        '''
        n = float(getattr(self.hp, 'align_prior_epochs', 2))
        if n <= 0:
            return 0.0
        return float(max(0.0, 1.0 - self.current_epoch / n))

    def forward(self, data_batch, batch_i, mode):
        ph_seqs, grp_seq, audio_batch, ph_seq_lens, wav_lens, boundaries = data_batch

        # Lengths first: they say how much of the padded clip is worth computing at all.
        spectral_lens = self.calc_spec_len(wav_lens)
        need_flux = bool(self.loss_fn.bonus_weight) and mode != 'predict'
        logits_c, logits_g, _, *flux = self.encode_clip(
            audio_batch, max_frames=int(spectral_lens.max()), return_flux=need_flux)
        flux = flux[0] if flux else None
        log_probs_c = F.log_softmax(logits_c.float(), dim=2)
        log_probs_g = F.log_softmax(logits_g.float(), dim=2)
        spectral_lens = spectral_lens.clamp(max=log_probs_c.shape[1])

        if mode == 'predict':
            return log_probs_c, log_probs_g

        loss, stats, alignment = self.loss_fn(
            log_probs_c, log_probs_g, ph_seqs, grp_seq, ph_seq_lens, spectral_lens,
            prior_weight=self.prior_weight_now() if mode == 'train' else 0.0, flux=flux)

        for key, value in stats.items():
            self.log(f'{mode}_{key}', value, prog_bar=False, on_step=False,
                     on_epoch=True, sync_dist=True)
        self.log(f'{mode}_loss', loss, prog_bar=(mode == 'train'), on_step=False,
                 on_epoch=True, sync_dist=True)
        if mode == 'train':
            # No `rank_zero_only` here even though the value is identical on every rank:
            # combining it with sync_dist means one rank enters a collective the others skip.
            # Every log in this package is plain `sync_dist=True`, with no exceptions.
            self.log('align_prior', self.prior_weight_now(), prog_bar=False, on_step=False,
                     on_epoch=True, sync_dist=True)
            return loss

        # --- validation bookkeeping ------------------------------------------------------
        # Only the first `per_capacity` clips of the epoch are scored for PER/GER, and only on
        # the epochs `per_epoch_active` allows. `MetricsCalculator5.calculate_per` costs
        # O(decoded_length * target_length) per clip in a Python double loop, and without a
        # blank class to collapse runs the decoded length starts near the FRAME COUNT rather
        # than near the phone count. Measured at 1992 frames per clip: 6.7 s/clip while the
        # argmax still flickers every frame, 0.68 s/clip at 50 ms runs, 0.14 s/clip at 250 ms
        # runs. So the bill shrinks by itself as the model sharpens - but the early epochs,
        # which are exactly the ones nobody wants to wait through, are the expensive ones.
        cum = self.metrics_cum.get(mode)
        if cum is None:                                  # per_enabled: false
            return loss
        room = int(cum.total_entries) - int(cum.idx)
        if self.per_epoch_active() and room > 0:
            n = min(room, log_probs_c.shape[0])
            target_bnd, pred_bnd = self._boundary_frames(boundaries, alignment, spectral_lens)
            cum.add_batch(
                log_probs_c[:n].detach().cpu(), ph_seqs[:n].detach().cpu(),
                ph_seq_lens[:n].detach().cpu(), log_probs_g[:n].detach().cpu(),
                grp_seq[:n].detach().cpu(),
                pred_lengths=spectral_lens[:n].detach().cpu(),
                target_boundaries=None if target_bnd is None else target_bnd[:n],
                pred_boundaries=None if pred_bnd is None else pred_bnd[:n])
        return loss

    def _boundary_frames(self, boundaries, alignment, spectral_lens):
        '''
        Labelled boundaries and the aligner's own onsets, both as frame indices, for
        `MetricsCalculator5.compute_boundary_metrics`.

        Only datasets that ship timestamps have any; everything else passes a zero tensor and
        the boundary block is skipped. `0` means padding in both tensors, which is also why the
        onset of the first phone (frame 0) is not reported - it is not a boundary anyway.
        '''
        if boundaries is None or boundaries.shape[1] <= 1 or int(boundaries.max()) == 0:
            return None, None
        self.has_boundaries = True
        target = self.samples_to_frames(boundaries.detach().cpu())

        onsets = alignment['onsets'].detach().cpu()          # [B, S_ext + 1]
        ext_lens = alignment['ext_lens'].detach().cpu()
        pred = torch.zeros_like(target)
        width = target.shape[1]
        for b in range(onsets.shape[0]):
            vals = [int(v) for v in onsets[b, 1:int(ext_lens[b])] if 0 < int(v) < int(spectral_lens[b])]
            vals = sorted(set(vals))[:width]
            if vals:
                pred[b, :len(vals)] = torch.tensor(vals)
        return target, pred

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def align_batch(self, log_probs, seqs, seq_lens, spectral_lens, flux=None):
        '''
        Force-align a whole batch. ONE DP run for the batch, not one per clip.

        This is the deliverable - the MFA-equivalent call - and it is the same DP the loss
        uses, so what comes out here is exactly what was trained, with no second
        implementation to drift out of step.

        Args:
            log_probs: [B, T, C] log posteriors
            seqs: [B, S] target phoneme ids, seq_lens: [B]
            spectral_lens: [B] valid frames per clip
            flux: [B, T] from `encode_clip(..., return_flux=True)`; required when
                `boundary_bonus_weight` is set
        Returns:
            list of B dicts, each with `phones` (the extended sequence, SIL pads included),
            `onsets` (integer frame per phone), `refined` (float frame, interpolated at the
            posterior crossing) and `used` (False for an optional SIL the path skipped). An
            empty dict marks a clip too short for its own sequence.
        '''
        # `loss_fn.align` IS the training-time aligner, so there is no second copy of the
        # decoder settings here to drift out of step with it.
        al = self.loss_fn.align(log_probs, seqs, seqs, seq_lens, spectral_lens, flux=flux)
        ext, ext_lens, feasible = al['ext'], al['ext_lens'], al['feasible']
        onsets = al['onsets'].cpu()

        # The refinement is a handful of scalar reads per boundary, so it goes to the CPU once
        # per batch rather than syncing with the GPU a few thousand times per epoch.
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

    @torch.no_grad()
    def align_clip_detail(self, log_probs, seq, n_frames, flux=None):
        '''Single-clip convenience wrapper over `align_batch`. `flux`: [T], see there.'''
        device = log_probs.device
        return self.align_batch(log_probs[:n_frames].unsqueeze(0), seq.to(device).unsqueeze(0),
                                torch.tensor([seq.numel()], device=device),
                                torch.tensor([n_frames], device=device),
                                flux=None if flux is None else flux[:n_frames].unsqueeze(0))[0]

    @torch.no_grad()
    def align_clip(self, log_probs, seq, n_frames, sub_frame=True, flux=None):
        '''
        Boundary times in MILLISECONDS: list of (phoneme_id, onset_ms), SIL included. The
        caller decides what counts as a segmentation point.
        '''
        d = self.align_clip_detail(log_probs, seq, n_frames, flux=flux)
        if not d:
            return []
        pos = d['refined'] if sub_frame else d['onsets']
        return [(ph, float(p) * self.ms_per_frame)
                for ph, p, used in zip(d['phones'], pos, d['used']) if used]

    # ------------------------------------------------------------------ lightning hooks

    def training_step(self, data_batch, batch_i):
        return self.forward(data_batch, batch_i, 'train')

    def validation_step(self, data_batch, batch_i):
        return self.forward(data_batch, batch_i, 'val')

    def test_step(self, data_batch, batch_i):
        return self.forward(data_batch, batch_i, 'test')

    def on_end_metrics_eval(self, mode):
        with torch.no_grad():
            # Nothing accumulated: this epoch skipped PER (`per_every_n_epochs`). Return
            # rather than compute - `calculate_per` over an empty buffer returns 1.0, which
            # would log a perfect-failure PER on every skipped epoch and look like a crash.
            if mode not in self.metrics_cum:                 # per_enabled: false
                return {}
            if int(self.metrics_cum[mode].idx) == 0:
                self.metrics_cum[mode].reset()
                return {}
            results = self.metrics_cum[mode].compute_sequential_metrics()
            if self.has_boundaries:
                results.update(self.metrics_cum[mode].compute_boundary_metrics())
                self.has_boundaries = False
            for k, v in results.items():
                self.log(f'{mode}_{k}', v, prog_bar=(k == 'PER'), on_step=False,
                         on_epoch=True, sync_dist=True)
            self.metrics_cum[mode].reset()
            if mode == 'test':
                self.metrics_cum[mode].clear_mem()
            return results

    def on_train_epoch_end(self):
        '''
        Per-rank GPU memory, logged so a divergence between ranks shows up on the dashboard
        instead of in someone's nvidia-smi.

        `allocated` is the real peak demand of the worst batch this epoch. `reserved` is what
        nvidia-smi shows: allocated plus whatever the caching allocator is holding but cannot
        reuse. The gap between them is fragmentation, and because each rank draws a different
        sequence of clip lengths, each rank strands a different amount - which is what makes
        two ranks on identical work report different numbers. Both are per-rank peaks, NOT
        reduced across ranks, so `sync_dist` would destroy the thing being measured; logging
        them from rank zero only is the point.
        '''
        if torch.cuda.is_available():
            gb = 2 ** 30
            alloc = torch.cuda.max_memory_allocated() / gb
            reserved = torch.cuda.max_memory_reserved() / gb
            self.log('gpu_peak_alloc_gb', alloc, prog_bar=False, on_step=False,
                     on_epoch=True, rank_zero_only=True)
            self.log('gpu_peak_reserved_gb', reserved, prog_bar=False, on_step=False,
                     on_epoch=True, rank_zero_only=True)
            self.log('gpu_frag_pct', 100.0 * (reserved - alloc) / max(alloc, 1e-6),
                     prog_bar=False, on_step=False, on_epoch=True, rank_zero_only=True)
            if self.main_node:
                print(f"[mem] rank {self.global_rank}: peak allocated {alloc:.1f} GB, "
                      f"reserved {reserved:.1f} GB ({100 * (reserved - alloc) / max(alloc, 1e-6):.0f}% held)")
            torch.cuda.reset_peak_memory_stats()   # per-epoch peaks, not run-to-date
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        epoch_time = (time.time() - self.last_epoch_time) / 60
        self.last_epoch_time = time.time()
        self.log("epoch_mins", epoch_time, prog_bar=False, on_step=False,
                 on_epoch=True, sync_dist=True)
        remaining = max(int(self.hp.epochs) - self.current_epoch, 0)
        self.log("hours_left", round(epoch_time * remaining / 60, 2), prog_bar=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        return self.on_end_metrics_eval('val')

    def on_test_epoch_end(self):
        return self.on_end_metrics_eval('test')

    def configure_optimizers(self):
        lr = float(self.hp.lr)
        optimizer = optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=lr, weight_decay=float(getattr(self.hp, 'weight_decay', 0.01)),
            betas=(0.9, 0.999), eps=1e-8)

        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        if steps_per_epoch < 1:
            raise ValueError("steps_per_epoch < 1: batch size too big or dataset too small")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=int(self.hp.epochs),
            steps_per_epoch=steps_per_epoch, pct_start=0.3, anneal_strategy='cos',
            div_factor=25.0, final_div_factor=1000.0)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}
