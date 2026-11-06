'''
p3cupe: Context-free Universal Phoneme Encoder, CNN only.

Takes a raw 16 kHz waveform window (120 ms by default) and emits one phoneme posterior per
5 ms frame. There is no transformer, no boundary head, no refiner, no quantizer and no blank
class - see `README.md` for why each of those went.

## Why CNN only

The trunk is a short-receptive-field CNN, and its receptive field is a printed, asserted number
rather than an emergent one. Capacity is a separate axis from time mixing: `FrameBlock`s add
per-frame depth without moving the receptive field when attention is disabled.

## Geometry (the numbers the rest of the package depends on)

    conv strides 4 * 4 * 5              = 80 samples  = 5.0 ms per frame
    120 ms window                       = 1920 samples = 24 frames
    receptive field (default context)   = 623 samples  = 38.9 ms, centred

At 5 ms, the frame grid gives a 2.5 ms half-frame quantisation scale, and the posterior-crossing
readout in `loss.py` can interpolate between frames.

Every conv has an ODD kernel and `padding = kernel // 2`, which is what makes frame `f` sit
centred on input sample `f * 80` with no offset correction. Keep it that way: an even kernel
anywhere in the stack shifts every timestamp this project produces by half a frame, and
nothing will raise.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    '''Conv -> BatchNorm -> GELU (-> Dropout), with a residual when shapes allow.'''

    def __init__(self, in_dim, out_dim, kernel, stride=1, dropout=0.0, residual=False):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError(f"kernels must be odd to keep frames centred, got {kernel}")
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel, stride=stride,
                              padding=kernel // 2, bias=False)
        self.norm = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = bool(residual) and (in_dim == out_dim) and (stride == 1)

    def forward(self, x):
        y = self.drop(self.act(self.norm(self.conv(x))))
        return x + y if self.residual else y


class LocalAttention(nn.Module):
    '''
    Self-attention restricted to a band of +-`radius` frames, with a learned relative-position
    bias per head (a band without position information cannot tell before from after, which
    is the one thing a boundary needs). Each layer adds 2 * radius frames to the receptive
    field, and `CupeCNN._receptive_field` counts it.
    '''

    def __init__(self, dim, heads, radius, dropout=0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"embed_dim {dim} is not divisible by attn_heads {heads}")
        self.heads, self.radius, self.dropout = int(heads), int(radius), float(dropout)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.rel_bias = nn.Parameter(torch.zeros(self.heads, 2 * self.radius + 1))

    def forward(self, x):
        n, t, d = x.shape
        q, k, v = self.qkv(x).view(n, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        rel = torch.arange(t, device=x.device)[None, :] - torch.arange(t, device=x.device)[:, None]
        bias = self.rel_bias[:, rel.clamp(-self.radius, self.radius) + self.radius]   # [H, T, T]
        bias = bias.masked_fill(rel.abs() > self.radius, float('-inf')).to(q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[None],
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(n, t, d))


class FrameBlock(nn.Module):
    '''
    Pre-norm residual block on the frame embeddings: [optional banded attention] then a
    per-frame MLP. With `attn_radius=0` it is a transformer block with the attention taken
    out - depth and width, and NO time mixing, so the receptive field is untouched.

    This block adds nonlinear per-frame capacity. With attention disabled it does not mix
    information between neighboring frames.

    The residual branches are zero-initialised, so every block starts as the identity and the
    trunk trains as the shallow model until the extra depth pays for itself.
    '''

    def __init__(self, dim, ffn_mult=4, attn_radius=0, attn_heads=8, dropout=0.1):
        super().__init__()
        self.attn = None
        if attn_radius > 0:
            self.attn_norm = nn.LayerNorm(dim)
            self.attn = LocalAttention(dim, attn_heads, attn_radius, dropout)
            nn.init.zeros_(self.attn.out.weight)
            nn.init.zeros_(self.attn.out.bias)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        '''x: [N, T, dim]'''
        if self.attn is not None:
            x = x + self.drop(self.attn(self.attn_norm(x)))
        return x + self.drop(self.ffn(self.ffn_norm(x)))


class CupeCNN(nn.Module):
    '''
    The model. `config` is the single source of truth for its shape and is what gets written
    into the checkpoint, so a reload rebuilds the same tensor sizes without consulting the yaml.

    Args (all also readable from `config`):
        input_wav_length: samples per window. Sets `frames_per_window`.
        cnn_channels: width of the first conv. Layers go C -> 2C -> embed_dim.
        embed_dim: per-frame embedding width, and the width of the context convs.
        context_kernels: stride-1 kernels applied after the downsampler. This list IS the
            receptive-field knob: each entry adds (k-1) * 80 samples = (k-1) * 5 ms. []
            gives a short field, [5, 3] gives 38.9 ms, and larger kernels give more context.
        phoneme_classes / phoneme_groups: 66 / 16 real labels. The heads emit one more of
            each because index 66 ('noise') and 16 ('noise') exist in the shared mapper and
            downstream tooling indexes by it. Nothing in this package ever makes them a
            target: there is no blank, and the slot simply stays unused.
        frame_blocks: number of `FrameBlock`s after the context convs - the CAPACITY knob,
            independent of the receptive field while `attn_radius` is 0. 0 = the original
            5.1M-param trunk (and what a checkpoint without this key rebuilds as).
        ffn_mult: hidden width of each block's MLP, as a multiple of embed_dim.
        attn_radius: 0 = no attention. >0 adds banded self-attention of +-radius frames to
            every block, and 2 * radius * 5 ms of receptive field PER BLOCK. Read
            `val_timit_dip_f1_25ms` before turning it on - it is the same trade the
            context_kernels note describes.
        local_delta: 0 = off. 1 feeds the phone classifier the embedding's own change to each
            neighbour, `[e_t - e_{t-1}, e_{t+1} - e_t]`, through a ZERO-INITIALISED projection
            added to the classifier's hidden layer. See `classify`.
    '''

    def __init__(self, input_wav_length=None, cnn_channels=192, embed_dim=512,
                 context_kernels=(5, 3), cnn_dropout=0.1, head_dropout=0.1,
                 noise_level=0.01, phoneme_classes=66, phoneme_groups=16,
                 frame_blocks=4, ffn_mult=4, attn_radius=0, attn_heads=8, block_dropout=0.1,
                 local_delta=0):
        super().__init__()
        self.config = None
        if input_wav_length is not None:
            self.config_model(input_wav_length, cnn_channels, embed_dim, context_kernels,
                              cnn_dropout, head_dropout, noise_level,
                              phoneme_classes, phoneme_groups, frame_blocks, ffn_mult,
                              attn_radius, attn_heads, block_dropout, local_delta)
            self.make_model()

    # ------------------------------------------------------------------ config

    def config_model(self, input_wav_length, cnn_channels, embed_dim, context_kernels,
                     cnn_dropout, head_dropout, noise_level, phoneme_classes, phoneme_groups,
                     frame_blocks=0, ffn_mult=4, attn_radius=0, attn_heads=8,
                     block_dropout=0.1, local_delta=0):
        self.config = {
            'input_wav_length': int(input_wav_length),
            'cnn_channels': int(cnn_channels),
            'embed_dim': int(embed_dim),
            'context_kernels': [int(k) for k in context_kernels],
            'cnn_dropout': float(cnn_dropout),
            'head_dropout': float(head_dropout),
            'noise_level': float(noise_level),
            'phoneme_classes': int(phoneme_classes),
            'phoneme_groups': int(phoneme_groups),
            'frame_blocks': int(frame_blocks),
            'ffn_mult': int(ffn_mult),
            'attn_radius': int(attn_radius),
            'attn_heads': int(attn_heads),
            'block_dropout': float(block_dropout),
            'local_delta': int(local_delta),
        }

    def load_config_state_dict(self, config_dict):
        self.config = dict(config_dict)

    def save_config_state_dict(self):
        return self.config

    # ------------------------------------------------------------------ build

    def make_model(self):
        cfg = self.config
        c = cfg['cnn_channels']
        d = cfg['embed_dim']
        drop = cfg['cnn_dropout']

        self.noise_level = cfg['noise_level']
        self.input_wav_length = int(cfg['input_wav_length'])
        self.classes_dim = cfg['phoneme_classes'] + 1      # +1: unused 'noise' slot, see docstring
        self.groups_dim = cfg['phoneme_groups'] + 1

        # Downsampler: 4 * 4 * 5 = 80 samples per frame = 5 ms at 16 kHz.
        self.feature_extractor = nn.Sequential(
            ConvBlock(1, c, kernel=15, stride=4, dropout=drop),
            ConvBlock(c, c * 2, kernel=9, stride=4, dropout=drop),
            ConvBlock(c * 2, d, kernel=7, stride=5, dropout=drop),
        )

        # Per-frame channel gate. It uses a pooled summary to rescale channels.
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d, d, 1, bias=True),
            nn.Sigmoid(),
        )

        # Stride-1 context. Residual, so the trunk can route around it if the extra span is
        # not paying for itself.
        self.context = nn.Sequential(*[
            ConvBlock(d, d, kernel=k, stride=1, dropout=drop, residual=True)
            for k in cfg['context_kernels']
        ])

        # Per-frame depth. `.get` with 0 so a checkpoint saved before these keys existed
        # rebuilds as exactly the network it was.
        self.attn_radius = int(cfg.get('attn_radius', 0))
        self.blocks = nn.Sequential(*[
            FrameBlock(d, ffn_mult=int(cfg.get('ffn_mult', 4)), attn_radius=self.attn_radius,
                       attn_heads=int(cfg.get('attn_heads', 8)),
                       dropout=float(cfg.get('block_dropout', 0.1)))
            for _ in range(int(cfg.get('frame_blocks', 0)))
        ])

        self.norm = nn.LayerNorm(d)

        self.classifier = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(cfg['head_dropout']),
            nn.Linear(d * 2, self.classes_dim),
        )
        self.group_classifier = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, self.groups_dim),
        )

        # Local-change branch into the phone classifier. Zero-initialised, so a model built
        # with it computes exactly what the checkpoint it was grown from computed, and the
        # warm start is an A/B against the same starting point. `.get` so older configs
        # rebuild without it.
        self.local_delta = int(cfg.get('local_delta', 0))
        self.delta_proj = None
        if self.local_delta:
            self.delta_proj = nn.Linear(2 * d, d * 2)
            nn.init.zeros_(self.delta_proj.weight)
            nn.init.zeros_(self.delta_proj.bias)

        self.frame_stride_samples = self._frame_stride()
        self.receptive_field_samples = self._receptive_field()
        # The delta reads one neighbour either side, so the phone POSTERIOR sees one frame
        # more each way than the embedding does. Reported separately: the trunk's number is
        # the one the self-test pins and the one the context experiments were about.
        self.head_receptive_field_samples = (self.receptive_field_samples
                                             + 2 * self.frame_stride_samples * bool(self.local_delta))
        self.frames_per_window = self.update_frames_per_window(self.input_wav_length)

    def _strided_convs(self):
        return [m for m in self.feature_extractor.modules() if isinstance(m, nn.Conv1d)]

    def _frame_stride(self):
        stride = 1
        for conv in self._strided_convs():
            stride *= int(conv.stride[0])
        return stride

    def _receptive_field(self):
        '''Receptive field of one output frame, in input samples.'''
        rf, jump = 1, 1
        for block in (*self.feature_extractor, *self.context):
            conv = block.conv
            rf += (int(conv.kernel_size[0]) - 1) * jump
            jump *= int(conv.stride[0])
        rf += 2 * self.attn_radius * jump * len(self.blocks)   # 0 unless attention is on
        return rf

    def update_frames_per_window(self, input_wav_length):
        '''
        Frames the trunk emits for a window of this length. Measured by a dry run rather than
        by a formula: the formula and the conv stack drifted apart once in this lineage and the
        mismatch was invisible for a whole training run.
        '''
        self.input_wav_length = int(input_wav_length)
        self.config['input_wav_length'] = self.input_wav_length
        was_training = self.training
        self.eval()
        with torch.no_grad():
            probe = torch.zeros(1, 1, self.input_wav_length,
                                device=next(self.parameters()).device)
            frames = self.feature_extractor(probe).shape[-1]
        if was_training:
            self.train()
        self.frames_per_window = int(frames)
        return self.frames_per_window


    def describe(self, sample_rate=16000):
        ms = 1000.0 / sample_rate
        return (f"CupeCNN: {self.frames_per_window} frames/window "
                f"({self.input_wav_length} samples, {self.input_wav_length * ms:.1f} ms), "
                f"frame stride {self.frame_stride_samples} samples "
                f"({self.frame_stride_samples * ms:.3f} ms), "
                f"receptive field {self.receptive_field_samples} samples "
                f"({self.receptive_field_samples * ms:.1f} ms)"
                + (f" + local delta -> classifier {self.head_receptive_field_samples * ms:.1f} ms"
                   if self.local_delta else "") + ", "
                f"{sum(p.numel() for p in self.parameters()) / 1e6:.2f}M params")

    # ------------------------------------------------------------------ forward

    def encode(self, x, return_tap=False):
        '''
        Args:
            x: [N, window_samples] flattened windows
            return_tap: also return the `feature_extractor` output (before the frequency gate),
                [N, frames_per_window, embed_dim]. It is the tap whose adjacent-frame flux
                carries the boundary signal the aligner's onset bonus reads: 8.9 ms receptive
                field, nothing mixed across time yet. Later taps measured no better than the
                posteriors themselves (see the yaml's boundary_bonus notes).
        Returns:
            [N, frames_per_window, embed_dim] (, tap)
        '''
        if self.training and self.noise_level > 0:
            x = x + torch.randn_like(x) * self.noise_level

        tap = self.feature_extractor(x.unsqueeze(1))
        h = tap * self.freq_attention(tap)
        h = self.context(h)
        emb = self.norm(self.blocks(h.transpose(1, 2)))
        return (emb, tap.transpose(1, 2)) if return_tap else emb

    def classify(self, embeddings):
        '''
        embeddings: [N, T, embed_dim] -> (class logits, group logits)

        ## local_delta

        `repr_flux_weight` put boundary information into the embedding (flux f1@25 above
        log-mel for the first time) and none of it reached the posteriors' confidence dip. The
        hat asks the classifier to be flat across a phone interior, i.e. invariant to
        within-phone change, and a per-frame MLP can only see change the trunk has already
        folded into the frame itself. This hands the head the change directly, so it can
        place a dip without the trunk having to move.

        Both one-sided differences, not the centred one: the centred difference cancels at
        the frame where the change happens, and the sign says which neighbour is changing.
        Computed inside the window with a zero at each edge - those frames get ~1e-3 of the
        stitch weight, so the overlapping window supplies them. The group head is left alone.
        '''
        if self.delta_proj is None:
            return self.classifier(embeddings), self.group_classifier(embeddings)
        e = embeddings
        step = e[:, 1:] - e[:, :-1]
        zero = torch.zeros_like(e[:, :1])
        delta = torch.cat([torch.cat([zero, step], 1), torch.cat([step, zero], 1)], -1)
        base = self.classifier[0](e)
        extra = self.delta_proj(delta)
        if self.training:
            # Diagnostic only: how much of the hidden pre-activation the branch carries. A
            # plain tensor attribute, so it never enters the state_dict.
            self.delta_share = (extra.detach().float().norm()
                                / base.detach().float().norm().clamp(min=1e-6))
        return self.classifier[1:](base + extra), self.group_classifier(e)

    def forward(self, x):
        return self.classify(self.encode(x))


class CupeExtractor(nn.Module):
    '''
    Load a trained p3cupe checkpoint for downstream use (embeddings or posteriors).

    Windowing is the CALLER's job here: this wraps the per-window trunk, so
    feed it `[N, input_wav_length]` windows and stitch with `windowing.stitch_windows` if you
    need clip-long output. `trainer.ComposedModel.encode_clip` does exactly that and is the
    reference implementation.
    '''

    def __init__(self, ckpt_path, device='cuda'):
        '''`ckpt_path`: a training checkpoint, a release directory written by
        `p3cupe/export_hf.py` (config.json + model.safetensors), or a Hugging Face repo id
        (e.g. "Tabahi/CUPE-3-en"; only those two files are downloaded).'''
        super().__init__()
        import os
        self.device = device
        if not os.path.exists(ckpt_path):
            from huggingface_hub import snapshot_download
            ckpt_path = snapshot_download(ckpt_path,
                                          allow_patterns=['config.json', 'model.safetensors'])
        self.config = None
        if os.path.isdir(ckpt_path):
            import json
            from safetensors.torch import load_file
            from p3cupe.export_hf import build_model
            with open(os.path.join(ckpt_path, 'config.json')) as f:
                self.config = json.load(f)
            model = build_model(self.config,
                                load_file(os.path.join(ckpt_path, 'model.safetensors')))
        else:
            model = self._from_checkpoint(ckpt_path)

        self.model = model.to(device).eval()
        self.frames_per_window = model.frames_per_window
        self.frame_stride_samples = model.frame_stride_samples
        print("p3cupe loaded:", model.describe())

    @staticmethod
    def _from_checkpoint(ckpt_path):
        checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        if 'model_config' not in checkpoint:
            raise ValueError(f"no model_config in {ckpt_path}; not a p3cupe checkpoint")

        model = CupeCNN()
        model.load_config_state_dict(checkpoint['model_config'])
        model.make_model()

        state = {k.replace('model.', '', 1): v for k, v in checkpoint['state_dict'].items()}
        keep = ('feature_extractor.', 'freq_attention.', 'context.', 'blocks.', 'norm.',
                'classifier.', 'group_classifier.', 'delta_proj.')
        state = {k: v for k, v in state.items() if k.startswith(keep)}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise ValueError(f"checkpoint is missing weights for: {sorted(missing)[:8]}")
        if unexpected:
            print(f"WARN: ignoring unexpected checkpoint keys: {sorted(unexpected)[:8]}")
        return model

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    @torch.no_grad()
    def forward(self, windows):
        return self.model.encode(windows.to(self.device))

    @torch.no_grad()
    def predict(self, windows, return_embeddings=False):
        emb = self.model.encode(windows.to(self.device))
        logits_class, logits_group = self.model.classify(emb)
        if return_embeddings:
            return logits_class, logits_group, emb
        return logits_class, logits_group


# Familiar alias for downstream callers.
ContextFreePhonemeRecognizer = CupeCNN


if __name__ == "__main__":
    torch.manual_seed(0)
    model = CupeCNN(input_wav_length=1920)
    print(model.describe())

    assert model.frame_stride_samples == 80, model.frame_stride_samples
    assert model.frames_per_window == 24, model.frames_per_window
    assert model.receptive_field_samples == 623, model.receptive_field_samples

    x = torch.randn(7, 1920)
    logits_c, logits_g = model(x)
    print("logits:", logits_c.shape, logits_g.shape)
    assert logits_c.shape == (7, 24, 67)
    assert logits_g.shape == (7, 24, 17)

    # Centring check: a click at sample 800 must move frame 800/80 = 10 and leave distant
    # frames alone. This is the assertion that catches an even kernel or a wrong padding.
    model.eval()
    with torch.no_grad():
        base = torch.zeros(1, 1920)
        emb0 = model.encode(base)
        spike = base.clone()
        spike[0, 800] = 5.0
        emb1 = model.encode(spike)
    delta = (emb1 - emb0).abs().mean(-1)[0]
    print("peak frame:", int(delta.argmax()), "(expected 10)")
    assert int(delta.argmax()) == 10, delta

    # Per-frame blocks must not mix time: frames beyond the conv receptive field stay put
    # even after the zero-initialised branches are given real weights. Not exactly zero:
    # `freq_attention` pools over the whole window, so every frame sees ~1e-4 of the click
    # through the gate. That leak predates the blocks and is 3+ orders below the peak.
    for p in model.blocks.parameters():
        torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad():
        delta = (model.encode(spike) - model.encode(base)).abs().mean(-1)[0]
    reach = (623 // 2) // 80 + 1        # frames the conv stack can see either side of 800
    assert int(delta.argmax()) == 10 and float(delta[:10 - reach].max()) < 1e-3 * float(delta.max()), delta

    # Attention: the receptive field grows by 2 * radius frames per block, and the printed
    # number is what the model really sees.
    attn = CupeCNN(input_wav_length=1920, frame_blocks=2, attn_radius=2).eval()
    assert attn.receptive_field_samples == 623 + 2 * 2 * 80 * 2, attn.receptive_field_samples
    for p in attn.blocks.parameters():
        torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad():
        delta = (attn.encode(spike) - attn.encode(base)).abs().mean(-1)[0]
    reach = (attn.receptive_field_samples // 2) // 80 + 1
    assert int(delta.argmax()) == 10 and float(delta[:10 - reach].max()) < 1e-3 * float(delta.max()), delta
    print(attn.describe())

    # A checkpoint config from before the blocks existed rebuilds as the same 5.1M network.
    legacy = {k: v for k, v in model.config.items()
              if k not in ('frame_blocks', 'ffn_mult', 'attn_radius', 'attn_heads',
                           'block_dropout', 'local_delta')}
    old = CupeCNN()
    old.load_config_state_dict(legacy)
    old.make_model()
    assert len(old.blocks) == 0 and old.receptive_field_samples == 623
    assert old.delta_proj is None
    print("legacy config:", old.describe())

    # local_delta: grown from a plain model's weights it must compute EXACTLY the same logits
    # (zero-initialised branch), and once trained its reach is one frame further each way.
    grown = CupeCNN(input_wav_length=1920, local_delta=1).eval()
    missing, unexpected = grown.load_state_dict(model.state_dict(), strict=False)
    assert sorted(missing) == ['delta_proj.bias', 'delta_proj.weight'] and not unexpected, missing
    assert grown.receptive_field_samples == 623 and grown.head_receptive_field_samples == 783
    with torch.no_grad():
        a, b = model(x.clone()), grown(x.clone())
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), "grown head must start identical"
    torch.nn.init.normal_(grown.delta_proj.weight, std=0.05)
    with torch.no_grad():
        g0 = grown.classify(grown.encode(base))[0]
        g1 = grown.classify(grown.encode(spike))[0]
        m0 = model.classify(model.encode(base))[0]
        m1 = model.classify(model.encode(spike))[0]
    reach_g = int(((g1 - g0).abs().mean(-1)[0] > 1e-3 * float((g1 - g0).abs().max())).sum())
    reach_m = int(((m1 - m0).abs().mean(-1)[0] > 1e-3 * float((m1 - m0).abs().max())).sum())
    assert reach_g == reach_m + 2, (reach_g, reach_m)
    print(grown.describe())
    print("model.py self-test OK")
