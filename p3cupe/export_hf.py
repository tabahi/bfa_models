'''
Export a p3cupe (CUPE v3) training checkpoint as a self-contained model directory, the same
layout as p4mbfa's releases:

    model.safetensors   CupeCNN weights only (no optimiser, no Lightning state)
    config.json         architecture, frame geometry, the decoder settings it is meant to be
                        aligned with, and the ph66 phoneme / group names by index
    README.md           a model card, only if the directory has none (never overwritten)

`model.CupeExtractor(<dir | HF repo id | ckpt>)` loads it. p3cupe is superseded by p4mbfa;
this exists to publish its English checkpoint, and is driven by p4mbfa's manifest:

    python p4mbfa/export_hf.py release      # entries with `suite: p3cupe` come through here

The export is verified before anything is written: the weights load strictly into a model built
from config.json.
'''

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from p3cupe.model import CupeCNN  # noqa: E402

FORMAT = 'p3cupe'
FORMAT_VERSION = 1
SAMPLE_RATE = 16000              # p3cupe never stored it; its trainer and data are 16 kHz only
GEOMETRY_KEYS = ('window_size_ms', 'window_stride_ms')
DECODER_KEYS = ('min_phone_frames', 'soft_phone_frames', 'soft_frame_cost', 'sil_skippable',
                'optional_sil', 'class_prior_norm', 'boundary_bonus_weight',
                'boundary_contrast_frames', 'boundary_flux_source')
PROVENANCE_KEYS = ('experiment', 'note', 'data_src', 'ckpt_path', 'min_phone_frames',
                   'ramp_frames', 'soft_target_mix', 'lr', 'epochs')
CODE_URL = 'https://github.com/tabahi/bfa_models'


def _ph66():
    from utils.ph66_mapper import phoneme_groups_index, phoneme_groups_mapper, \
        phoneme_mapped_index
    def by_index(table, what):
        names = [None] * len(table)
        for name, i in table.items():
            if not 0 <= i < len(table) or names[i] is not None:
                raise ValueError(f"ph66 {what} index {i} ('{name}') is out of range or duplicated")
            names[i] = name
        return names
    phonemes = by_index(phoneme_mapped_index, 'phoneme')
    groups = by_index(phoneme_groups_index, 'group')
    to_group = [phoneme_groups_mapper[i] for i in range(len(phonemes))]
    return phonemes, groups, to_group


def build_release_config(ckpt, source_name, decoder_overrides, selection=None):
    '''config.json contents. `decoder_overrides` replace the checkpoint's training-time decoder
    settings (p3cupe checkpoints are re-decoded better than they were trained, see the card).'''
    if 'model_config' not in ckpt or 'phoneme_classes' not in ckpt['model_config']:
        raise ValueError(f"{source_name}: not a p3cupe checkpoint (no p3cupe model_config)")
    cfg = dict(ckpt['model_config'])
    hp = dict(ckpt['hyper_parameters'])
    missing = [k for k in (*GEOMETRY_KEYS, *DECODER_KEYS) if k not in hp]
    if missing:
        raise KeyError(f"{source_name}: hyper_parameters lack {missing}")
    unknown = [k for k in decoder_overrides if k not in DECODER_KEYS]
    if unknown:
        raise KeyError(f"decoder_overrides {unknown} are not decoder settings {DECODER_KEYS}")

    phonemes, groups, to_group = _ph66()
    sizes = {'phonemes': (cfg['phoneme_classes'] + 1, len(phonemes)),
             'groups': (cfg['phoneme_groups'] + 1, len(groups))}
    bad = {k: v for k, v in sizes.items() if v[0] != v[1]}
    if bad:
        raise ValueError(f"{source_name}: head sizes disagree with utils/ph66_mapper.py "
                         f"(checkpoint, table): {bad}")

    training = {**{k: hp.get(k) for k in PROVENANCE_KEYS},
                'decoder_at_training': {k: hp[k] for k in DECODER_KEYS},
                'epoch': ckpt.get('epoch'), 'global_step': ckpt.get('global_step'),
                'source_checkpoint': os.path.basename(source_name)}
    if selection is not None:
        training['selection'] = dict(selection)
    return {
        'format': FORMAT,
        'format_version': FORMAT_VERSION,
        'model_config': cfg,
        'geometry': {'sample_rate': SAMPLE_RATE, **{k: hp[k] for k in GEOMETRY_KEYS}},
        'decoder': {k: decoder_overrides.get(k, hp[k]) for k in DECODER_KEYS},
        'labels': {'phonemes': phonemes, 'phoneme_groups': groups,
                   'phoneme_to_group': to_group, 'sil': 0},
        'training': training,
    }


def model_weights(ckpt, source_name):
    sd = ckpt['state_dict']
    stray = [k for k in sd if not k.startswith('model.')]
    if stray:
        raise ValueError(f"{source_name}: tensors outside the model: {stray[:8]}")
    return {k[len('model.'):]: v.detach().cpu().contiguous() for k, v in sd.items()}


def build_model(config, state):
    '''CupeCNN from a release config, with `state` loaded strictly.'''
    if config.get('format') != FORMAT or int(config.get('format_version', -1)) != FORMAT_VERSION:
        raise ValueError(f"not a {FORMAT} v{FORMAT_VERSION} config: "
                         f"{config.get('format')} v{config.get('format_version')}")
    model = CupeCNN()
    model.load_config_state_dict(config['model_config'])
    model.make_model()
    model.load_state_dict(state, strict=True)
    return model


def model_card(config, trained_langs, repo_id, ckpt_name):
    train, cfg, geo, dec = (config['training'], config['model_config'], config['geometry'],
                            config['decoder'])
    sel = train['selection']
    rows = "\n".join(f"| `{k}` | {v} |" for k, v in sel['metrics'].items())
    stride = round(geo['window_stride_ms'] * geo['sample_rate'] / 1000)
    front = "\n".join(["---", "license: agpl-3.0", "library_name: pytorch", "language:",
                       *[f"- {l.split('-')[0]}" for l in trained_langs],
                       "tags:", "- forced-alignment", "- phoneme-recognition",
                       "- phoneme-segmentation", "- speech", "---", ""])
    return front + f'''
# CUPE v3 (p3cupe) · English

A context-free CNN phoneme encoder (the third CUPE), trained without boundary labels to place
phoneme boundaries: one ph66 phoneme distribution every 5 ms, each frame seeing at most
{1000 * cfg['input_wav_length'] / geo['sample_rate']:.0f} ms of audio (38.9 ms receptive field). Boundaries come from a segmental Viterbi
decoder over the known phone sequence, read at the crossing of neighbouring posteriors.

**Superseded by p4mbfa** (`Tabahi/mbfa-<language group>`, standard_g2p labels, many languages).
Kept for BFA's ph66 pipeline.

| head | labels | size |
|---|---|---|
| phonemes | ph66 (`config.json` labels.phonemes, index 0 = SIL; the last slot, 'noise', is never trained) | {cfg['phoneme_classes'] + 1} |
| groups | ph66 phoneme groups (labels.phoneme_groups) | {cfg['phoneme_groups'] + 1} |

## Training

- **Data:** {sel['train_data']}.
- **Checkpoint:** experiment `{train['experiment']}`, epoch {train['epoch']}, selected on
  {sel['selected_on']}.

{sel['note']}

| metric | value |
|---|---|
{rows}

TIMIT: 400 clips (English, never trained on), split in a `tune` half (picked the checkpoint and
the decoder) and a `holdout` half. `lev_f1_25ms` is phone-boundary F1 within 25 ms;
`phone_hit_25ms` is the share of phones whose onset is within 25 ms. Scored with the decoder in
`config.json` ("decoder": min {dec['min_phone_frames']} + soft {dec['soft_phone_frames']} frames,
log-mel onset bonus {dec['boundary_bonus_weight']}); `bonus0` = the same decoder without the bonus.
The settings it was trained with are under training.decoder_at_training.

## Usage

The code is in [{CODE_URL}]({CODE_URL}) (`p3cupe/`); clone it and run from its root.

```python
import torch
from p3cupe.model import CupeExtractor
from p3cupe.windowing import slice_windows, stitch_windows

ext = CupeExtractor("{repo_id}", device="cpu")         # or a release dir / checkpoint
wav = torch.randn(1, 1, 3 * 16000)                     # [B, 1, samples] at 16 kHz
wav = wav / wav.pow(2).mean().sqrt()                   # unit RMS per clip, as in training
win = slice_windows(wav, {geo['sample_rate']}, {geo['window_size_ms']}, {geo['window_stride_ms']})  # [B, n_windows, {cfg['input_wav_length']}]
b, n, _ = win.shape
logits_ph, logits_grp = ext.predict(win.reshape(b * n, -1))
stride_frames = {stride} // ext.frame_stride_samples
ph = stitch_windows(logits_ph.reshape(b, n, *logits_ph.shape[1:]), n,
                    ext.frames_per_window, stride_frames)  # [B, frames, {cfg['phoneme_classes'] + 1}], 5 ms frames
```

Forced alignment over a known ph66 sequence is `p3cupe.loss.PeakLoss.align` with the
`config.json` decoder settings, as `p3cupe/timit_probe.py` does.

## Training checkpoint

`ckpt/{ckpt_name}` is the PyTorch Lightning checkpoint (weights only; a pickle, so load it only
if you trust this repo), for `ckpt_path:` in `p3cupe/cupe_hp.yaml` or `trunk_init_path:` in
p4mbfa.

License: AGPL-3.0.
'''


def export(ckpt_path, out_dir, decoder_overrides, trained_langs, selection, repo_id,
           ckpt_name):
    import json
    from safetensors.torch import save_file
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = build_release_config(ckpt, ckpt_path, decoder_overrides, selection)
    state = model_weights(ckpt, ckpt_path)
    build_model(config, state)                         # verify before writing anything

    os.makedirs(out_dir, exist_ok=True)
    save_file(state, os.path.join(out_dir, 'model.safetensors'), metadata={'format': 'pt'})
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    card = os.path.join(out_dir, 'README.md')
    card_note = "kept the existing README.md"
    if not os.path.exists(card):
        with open(card, 'w') as f:
            f.write(model_card(config, trained_langs, repo_id, ckpt_name))
        card_note = "wrote a README.md model card"
    n = sum(v.numel() for v in state.values())
    print(f"Exported {ckpt_path}\n   -> {out_dir}: {len(state)} tensors, {n / 1e6:.2f}M params, "
          f"p3cupe ph66; {card_note}")
    return config
