'''
Lightning module for p4mbfa.

Its whole job: slice a clip into context-free windows, run the CNN on every window, stitch the
per-window outputs back onto one 5 ms frame grid, and hand the three heads' log posteriors to
`PeakLoss`. Externally it behaves as if the model had processed the whole clip; internally
nothing ever saw more than 120 ms.

## What came over from p3cupe, and what did not

Carried over unchanged: windowing and stitching, the segmental Viterbi aligner and its
decoder knobs (min + soft duration, skippable SIL, onset bonus from mel or tap flux, class-prior
normalisation), the diagonal alignment prior, soft onsets, the OneCycle schedule.

Left behind, because p3cupe's final recipe (pc01a) had every one of them switched off after
measuring it: the EMA teacher (`ema_decay`), warped views (`aug_views`), the representation-flux
term (`repr_flux_weight`), the local-delta branch (`local_delta`, and with it
`warm_start_arch_overrides`), `soft_target_mix`, and PER/GER (`per_enabled`); the identity guards
are now `*_frame_acc`.

## Checkpoints and the replaceable heads

`on_load_checkpoint` is strict: the checkpoint's trunk shape must equal the yaml's, and its
`phg_head` must match `n_phg`. The fine heads (`ph_head`, `tone_head`) then either:

    reset_fine_heads: false   must match the yaml's lang_group / n_tokens / n_tones, or raise
    reset_fine_heads: true    are discarded and rebuilt fresh for the yaml's lang_group /
                              n_tokens / n_tones; trunk and phg_head are kept

A p3cupe checkpoint is not a p4mbfa checkpoint (its heads are ph66 / pg16): load its trunk with
`trunk_init_path` instead, which leaves all three heads freshly initialised.
'''

import time

import lightning as L
import torch
import torch.nn.functional as F
from argparse import Namespace
from torch import optim

from p4mbfa import clip
from p4mbfa.loss import PeakLoss
from p4mbfa.model import (FINE_HEAD_KEYS, FINE_HEAD_PREFIXES, REGULARISATION_KEYS,
                          TRUNK_PREFIXES, TRUNK_SHAPE_KEYS, MbfaCNN)


class ComposedModel(L.LightningModule):

    def __init__(self, hp=None):
        super().__init__()
        self.model_ready = False
        self.main_node = True
        self.last_epoch_time = time.time()

        if hp is not None:
            if isinstance(hp, dict):
                hp = Namespace(**hp)
            self.hp = hp
            self.save_hyperparameters(self.hp)
            self.build_model()

    # ------------------------------------------------------------------ build

    def build_model(self):
        print("............................. building model")
        self.sample_rate = int(self.hp.sample_rate)
        self.window_size_ms = float(self.hp.window_size_ms)
        self.stride_ms = float(self.hp.window_stride_ms)
        assert self.window_size_ms > self.stride_ms, "windows must overlap"

        self.window_size_wav = int(round(self.window_size_ms * self.sample_rate / 1000))
        self.stride_size_wav = int(round(self.stride_ms * self.sample_rate / 1000))
        self.fixed_window_batch = bool(self.hp.fixed_window_batch)

        self.model = MbfaCNN(
            input_wav_length=self.window_size_wav,
            cnn_channels=int(self.hp.cnn_channels),
            embed_dim=int(self.hp.embed_dim),
            context_kernels=list(self.hp.context_kernels),
            cnn_dropout=float(self.hp.cnn_dropout),
            head_dropout=float(self.hp.head_dropout),
            noise_level=float(self.hp.noise_level),
            frame_blocks=int(self.hp.frame_blocks),
            ffn_mult=int(self.hp.ffn_mult),
            attn_radius=int(self.hp.attn_radius),
            attn_heads=int(self.hp.attn_heads),
            block_dropout=float(self.hp.block_dropout),
            n_phg=int(self.hp.n_phg),
            lang_group=str(self.hp.lang_group),
            n_tokens=int(self.hp.n_tokens),
            n_tones=int(self.hp.n_tones),
        )

        self._resolve_geometry()
        self.loss_fn = PeakLoss()
        self._read_loss_hp()

        self.params_count = sum(p.numel() for p in self.parameters())
        print(self.model.describe(self.sample_rate))
        print("............................. model built.")
        self.model_ready = True
        return True

    def _read_loss_hp(self):
        '''
        Every loss and decoder knob, read from the LIVE hp - at build and again in
        `configure()`, so a warm start trains with the yaml's settings, not the checkpoint's.
        '''
        hp, lf = self.hp, self.loss_fn
        lf.min_frames = int(hp.min_phone_frames)
        lf.ramp_frames = float(hp.ramp_frames)
        lf.group_weight = float(hp.group_loss_weight)
        lf.tone_weight = float(hp.tone_loss_weight)
        lf.optional_sil = bool(hp.optional_sil)
        lf.prior_sigma = float(hp.align_prior_sigma)
        lf.prior_strength = float(hp.align_prior_strength)
        lf.class_prior_norm = bool(hp.class_prior_norm)
        lf.soft_frames = int(hp.soft_phone_frames)
        lf.soft_cost = float(hp.soft_frame_cost)
        lf.sil_skippable = bool(hp.sil_skippable)
        lf.bonus_weight = float(hp.boundary_bonus_weight)
        lf.contrast_frames = int(hp.boundary_contrast_frames)
        lf.soft_target_scale = float(hp.soft_target_scale)
        lf.soft_onset_mix = float(hp.soft_onset_mix)
        lf.soft_onset_ramp_k = float(hp.soft_onset_ramp_k)
        r_max = hp.soft_onset_ramp_max
        lf.soft_onset_ramp_max = None if r_max is None else float(r_max)
        if hp.boundary_flux_source not in ('mel', 'tap'):
            raise ValueError(f"boundary_flux_source must be 'mel' or 'tap', "
                             f"got {hp.boundary_flux_source!r}")

    def _resolve_geometry(self):
        '''Pin down the frame grid, and refuse to run on one that does not divide evenly.'''
        self.geometry = clip.ClipGeometry.for_model(self.model, self.sample_rate,
                                                    self.window_size_ms, self.stride_ms)
        self.frames_per_window = self.geometry.frames_per_window
        self.frame_stride_samples = self.geometry.frame_stride_samples
        self.stride_frames = self.geometry.stride_frames
        self.ms_per_frame = self.geometry.ms_per_frame
        print(f"Geometry: {self.ms_per_frame:.4f} ms/frame, {self.frames_per_window} "
              f"frames/window, window advance {self.stride_frames} frames "
              f"({self.stride_ms} ms), receptive field "
              f"{1000.0 * self.model.receptive_field_samples / self.sample_rate:.1f} ms")

    def update_hyperparams(self, hp):
        '''Re-read the live yaml after a warm start (logging and loss settings).'''
        self.hp = hp
        self.save_hyperparameters(self.hp)
        self.last_epoch_time = time.time()

    # ------------------------------------------------------------------ checkpoints

    def on_save_checkpoint(self, checkpoint):
        checkpoint['model_config'] = self.model.save_config_state_dict()
        return checkpoint

    def _check_trunk_config(self, saved, source):
        '''
        The saved trunk shape must equal the one just built from the yaml, with one exception:
        with `grow_frame_blocks: true` the yaml may have MORE `frame_blocks` than the
        checkpoint. A FrameBlock's residual branch is zero-initialised, so the added blocks
        start as the identity and the grown model computes exactly what the checkpoint did.

        Returns the checkpoint's block count; blocks at or past it keep their fresh init.
        '''
        differing = {k: (saved.get(k), self.model.config[k]) for k in TRUNK_SHAPE_KEYS
                     if saved.get(k) != self.model.config[k]}
        n_saved = int(saved.get('frame_blocks', self.model.config['frame_blocks']))
        n_mine = int(self.model.config['frame_blocks'])
        if 'frame_blocks' in differing and n_saved < n_mine:
            if not bool(self.hp.grow_frame_blocks):
                raise ValueError(
                    f"{source}: checkpoint has {n_saved} frame_blocks, the yaml {n_mine}. Set "
                    f"grow_frame_blocks: true to add {n_mine - n_saved} identity-initialised "
                    f"blocks, or set frame_blocks: {n_saved}.")
            del differing['frame_blocks']
            print(f"   grow_frame_blocks: blocks 0..{n_saved - 1} from the checkpoint, "
                  f"{n_saved}..{n_mine - 1} added as the identity (zero-initialised residual)")
        if differing:
            raise ValueError(
                f"{source}: trunk architecture differs from the yaml (checkpoint, yaml): "
                + ", ".join(f"{k}: {v}" for k, v in differing.items())
                + ". Set the yaml to the checkpoint's values.")
        for k in REGULARISATION_KEYS:
            if k in saved and saved[k] != self.model.config[k]:
                print(f"   {k}: checkpoint {saved[k]} -> yaml {self.model.config[k]} (yaml used)")
        return n_saved

    def _added_block_keys(self, n_saved):
        '''state_dict keys (relative to `self.model`) of the blocks a growth added.'''
        return {k for k in self.model.state_dict()
                if k.startswith('blocks.') and int(k.split('.')[1]) >= n_saved}

    def on_load_checkpoint(self, checkpoint):
        '''
        `load_from_checkpoint(path, hp=live_hp)` has already built the model from the LIVE
        yaml. This checks the checkpoint against it and, if `reset_fine_heads` is set, swaps the
        checkpoint's fine-head tensors for the fresh ones so the strict load that follows
        succeeds. Nothing is rebuilt from the checkpoint's own hyperparameters.
        '''
        saved = checkpoint.get('model_config')
        if saved is None or 'lang_group' not in saved:
            raise ValueError("checkpoint has no p4mbfa model_config (a p3cupe checkpoint?). "
                             "Load a p3cupe trunk with trunk_init_path instead of ckpt_path.")
        print("Loading from checkpoint")
        n_saved = self._check_trunk_config(saved, "ckpt_path")
        if int(saved['n_phg']) != self.model.n_phg:
            raise ValueError(f"checkpoint phg_head has {saved['n_phg']} groups, yaml n_phg is "
                             f"{self.model.n_phg}; the phg head is fixed and cannot be reset")

        state = checkpoint['state_dict']
        own = self.state_dict()
        for k in self._added_block_keys(n_saved):
            if 'model.' + k in state:
                raise ValueError(f"checkpoint already has {k}; its model_config is inconsistent")
            state['model.' + k] = own['model.' + k]
        mine = {k: self.model.config[k] for k in FINE_HEAD_KEYS}
        theirs = {k: saved[k] for k in FINE_HEAD_KEYS}
        if bool(self.hp.reset_fine_heads):
            fine = tuple('model.' + p for p in FINE_HEAD_PREFIXES)
            dropped = [k for k in state if k.startswith(fine)]
            for k in dropped:
                del state[k]
            for k in own:
                if k.startswith(fine):
                    state[k] = own[k]
            print(f"   reset_fine_heads: DISCARDED the checkpoint's ph_head/tone_head "
                  f"{theirs} ({len(dropped)} tensors); fresh heads for {mine}")
        elif mine != theirs:
            raise ValueError(f"checkpoint fine heads are {theirs} but the yaml asks for {mine}. "
                             f"Set reset_fine_heads: true to discard them and train new ones.")
        else:
            print(f"   fine heads kept: {theirs}")

    def load_trunk(self, path):
        '''
        Initialise the TRUNK only from a p3cupe or p4mbfa checkpoint; every head keeps its fresh
        initialisation. The trunk shape must match (see `_check_trunk_config` for the one
        exception, `grow_frame_blocks`), and every trunk tensor must be present - a partial trunk
        raises.
        '''
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        saved = ckpt.get('model_config')
        if saved is None:
            raise ValueError(f"{path} has no model_config")
        n_saved = self._check_trunk_config(saved, "trunk_init_path")
        prefixes = tuple('model.' + p for p in TRUNK_PREFIXES)
        trunk = {k[len('model.'):]: v for k, v in ckpt['state_dict'].items()
                 if k.startswith(prefixes)}
        want = {k for k in self.model.state_dict() if k.startswith(TRUNK_PREFIXES)}
        want -= self._added_block_keys(n_saved)
        missing, extra = sorted(want - set(trunk)), sorted(set(trunk) - want)
        if missing or extra:
            raise ValueError(f"{path}: trunk tensors missing {missing[:8]}, unexpected "
                             f"{extra[:8]}")
        result = self.model.load_state_dict(trunk, strict=False)
        added = self._added_block_keys(n_saved)
        not_heads = [k for k in result.missing_keys
                     if not k.startswith(('phg_head.', *FINE_HEAD_PREFIXES)) and k not in added]
        if not_heads or result.unexpected_keys:
            raise ValueError(f"trunk load left {not_heads[:8]} / {result.unexpected_keys[:8]}")
        print(f"Trunk initialised from {path}: {len(trunk)} tensors; phg_head, ph_head and "
              f"tone_head are freshly initialised")

    # ------------------------------------------------------------------ configure

    def configure(self, ph_seq_max, train_wav_len_max, val_wav_len_max=None,
                  train_total=0, val_total=0, main_node=True):
        print("............................. configure")
        self.main_node = main_node
        self.wav_len_max = int(train_wav_len_max)
        self._read_loss_hp()

        max_spec_len = int(self.calc_spec_len(torch.tensor([train_wav_len_max])).item())
        print(f"wav_len_max: {train_wav_len_max} samples "
              f"({train_wav_len_max / self.sample_rate:.1f} s) -> {max_spec_len} frames; "
              f"ph_seq_max: {ph_seq_max}; train {train_total} / val {val_total} segments")
        if self.loss_fn.soft_onset_mix == 0 and self.loss_fn.soft_onset_ramp_k > 0:
            raise ValueError("soft_onset_ramp_k needs soft_onset_mix > 0 (it widens ramps by the "
                             "occupancy's spread, which is only computed then)")
        print("------------------------------------------------------------------")
        return self.model

    # ------------------------------------------------------------------ geometry helpers

    def calc_spec_len(self, wav_lens):
        '''Valid stitched frames per clip, from the real sample lengths.'''
        return self.geometry.spectral_lengths(wav_lens)

    # ------------------------------------------------------------------ forward

    def encode_clip(self, audio_batch, max_frames=None, return_embeddings=False,
                    return_flux=False):
        '''
        Windowed forward over a whole clip; see `clip.encode_clip`, which this configures from
        the yaml (`boundary_flux_source`, `fixed_window_batch`).

        Returns:
            (logits_ph, logits_phg, logits_tone, embeddings|None[, flux]), each [B, T_full, ...]
        '''
        # The mel transform is cached in a plain dict outside the module tree: registered, its
        # buffers would enter the state_dict.
        cache = self.__dict__.setdefault('_mel_cache', {})
        return clip.encode_clip(self.model, self.geometry, audio_batch,
                                flux_source=str(self.hp.boundary_flux_source),
                                max_frames=max_frames,
                                fixed_window_batch=self.fixed_window_batch,
                                return_embeddings=return_embeddings, return_flux=return_flux,
                                cache=cache)

    def prior_weight_now(self):
        '''
        Diagonal-prior weight for the current epoch: 1 -> 0 linearly over
        `align_prior_epochs`, then off. The flat start: an alignment read off freshly
        initialised heads is noise, so it is pulled towards even durations until they are not.
        '''
        n = float(self.hp.align_prior_epochs)
        if n <= 0:
            return 0.0
        return float(max(0.0, 1.0 - self.current_epoch / n))

    def forward(self, data_batch, batch_i, mode):
        ph_seqs, grp_seqs, tone_seqs, tone_mask, audio_batch, ph_seq_lens, wav_lens = data_batch

        spectral_lens = self.calc_spec_len(wav_lens)
        need_flux = bool(self.loss_fn.bonus_weight)
        logits_ph, logits_phg, logits_tone, _, *flux = self.encode_clip(
            audio_batch, max_frames=int(spectral_lens.max()), return_flux=need_flux)
        flux = flux[0] if flux else None
        log_probs = F.log_softmax(logits_ph.float(), dim=2)
        log_probs_g = F.log_softmax(logits_phg.float(), dim=2)
        log_probs_t = F.log_softmax(logits_tone.float(), dim=2)
        spectral_lens = spectral_lens.clamp(max=log_probs.shape[1])

        loss, stats, _ = self.loss_fn(
            log_probs, log_probs_g, log_probs_t, ph_seqs, grp_seqs, tone_seqs, tone_mask,
            ph_seq_lens, spectral_lens,
            prior_weight=self.prior_weight_now() if mode == 'train' else 0.0, flux=flux)

        for key, value in stats.items():
            self.log(f'{mode}_{key}', value, prog_bar=False, on_step=False,
                     on_epoch=True, sync_dist=True)
        self.log(f'{mode}_loss', loss, prog_bar=(mode == 'train'), on_step=False,
                 on_epoch=True, sync_dist=True)
        if mode == 'train':
            self.log('align_prior', self.prior_weight_now(), prog_bar=False, on_step=False,
                     on_epoch=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def align_batch(self, log_probs, seqs, seq_lens, spectral_lens, flux=None):
        '''Force-align a whole batch with the training-time decoder; see
        `clip.decode_alignments`.'''
        return clip.decode_alignments(self.loss_fn, log_probs, seqs, seq_lens, spectral_lens,
                                      flux=flux)

    @torch.no_grad()
    def align_clip(self, log_probs, seq, n_frames, sub_frame=True, flux=None):
        '''Boundary times in MILLISECONDS: list of (token_id, onset_ms), SIL included.'''
        device = log_probs.device
        d = self.align_batch(log_probs[:n_frames].unsqueeze(0), seq.to(device).unsqueeze(0),
                             torch.tensor([seq.numel()], device=device),
                             torch.tensor([n_frames], device=device),
                             flux=None if flux is None else flux[:n_frames].unsqueeze(0))[0]
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

    def on_train_epoch_end(self):
        '''Per-rank peak GPU memory, printed (not synced: the per-rank values are the point).'''
        if torch.cuda.is_available():
            gb = 2 ** 30
            alloc = torch.cuda.max_memory_allocated() / gb
            reserved = torch.cuda.max_memory_reserved() / gb
            if self.main_node:
                print(f"[mem] rank {self.global_rank}: peak allocated {alloc:.1f} GB, "
                      f"reserved {reserved:.1f} GB")
            torch.cuda.reset_peak_memory_stats()
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        epoch_time = (time.time() - self.last_epoch_time) / 60
        self.last_epoch_time = time.time()
        self.log("epoch_mins", epoch_time, prog_bar=False, on_step=False,
                 on_epoch=True, sync_dist=True)
        remaining = max(int(self.hp.epochs) - self.current_epoch, 0)
        self.log("hours_left", round(epoch_time * remaining / 60, 2), prog_bar=True,
                 on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        lr = float(self.hp.lr)
        optimizer = optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=lr, weight_decay=float(self.hp.weight_decay), betas=(0.9, 0.999), eps=1e-8)

        # Lightning steps the scheduler once per OPTIMISER step, so the cycle length is
        # batches / accumulate_grad_batches (p3cupe pb01a measured the cost of getting this
        # wrong: the anneal never happened).
        accum = max(1, int(self.trainer.accumulate_grad_batches))
        batches = len(self.trainer.datamodule.train_dataloader())
        steps_per_epoch = -(-batches // accum)
        if steps_per_epoch < 1:
            raise ValueError("steps_per_epoch < 1: batch size too big or dataset too small")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=int(self.hp.epochs),
            steps_per_epoch=steps_per_epoch, pct_start=0.3, anneal_strategy='cos',
            div_factor=25.0, final_div_factor=1000.0)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}

