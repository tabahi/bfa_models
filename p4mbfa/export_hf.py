'''
Export p4mbfa training checkpoints as self-contained model directories (Hugging Face layout),
and push them to the Hub.

    python p4mbfa/export_hf.py export <checkpoint> <out_dir> [--trained-langs "en de fr ..."]
    python p4mbfa/export_hf.py release [--manifest p4mbfa/hf_release.yaml] [--only NAME ...]
    python p4mbfa/export_hf.py push    [--manifest p4mbfa/hf_release.yaml] [--only NAME ...]
                                       [--dry-run] [--private] [--message "..."]

`export` converts one checkpoint. `release` builds every model listed in hf_release.yaml (which
checkpoint, why it was chosen, its metrics) into `<out_dir>/<name>/`, and `push` uploads each
of those folders to `<namespace>/<name>` (creating the repo if needed). `push` needs a token
with write access (`hf auth login`) and checks every folder against the manifest first.
Entries with `suite: p3cupe` (the English CUPE v3) are exported by `p3cupe/export_hf.py` into
the same folder layout and pushed the same way.

A release directory holds:
    model.safetensors   MbfaCNN weights only (no optimiser, no Lightning state)
    config.json         everything inference needs, so a published model loads without the
                        training yaml or standard_g2p: architecture, frame geometry, the decoder
                        settings it was trained with, and its label spaces as token strings
    README.md           a model card, only if the directory has none (never overwritten)
    ckpt/<name>.ckpt    (`release` only) the training checkpoint, for `ckpt_path:` fine-tuning

`inference.MbfaAligner.from_pretrained(dir | repo id)` loads it. The export is verified before
anything is written: the weights load strictly into a model built from config.json.

Exporting reads the label strings from standard_g2p (`standard_g2p_repo` in .local_paths.yaml),
the same tables the training labels came from, and raises if their sizes disagree with the
checkpoint's heads.
'''

import argparse
import json
import os
import shutil
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from p4mbfa.model import MbfaCNN  # noqa: E402

FORMAT = 'p4mbfa'
FORMAT_VERSION = 1
SPECIAL_TOKENS = {'<blank>': 0, 'SIL': 1, 'noise': 2, '<unk>': 3}
# Training hyperparameters inference needs, copied verbatim into config.json.
GEOMETRY_KEYS = ('sample_rate', 'window_size_ms', 'window_stride_ms')
DECODER_KEYS = ('min_phone_frames', 'soft_phone_frames', 'soft_frame_cost', 'sil_skippable',
                'optional_sil', 'class_prior_norm', 'boundary_bonus_weight',
                'boundary_contrast_frames', 'boundary_flux_source')
PROVENANCE_KEYS = ('experiment', 'note', 'data_src', 'train_limit', 'val_limit',
                   'exclude_langs', 'duration_max', 'ckpt_path', 'trunk_init_path',
                   'reset_fine_heads', 'ramp_frames', 'soft_onset_mix', 'noise_level', 'lr',
                   'epochs')
RELEASE_FILES = ('config.json', 'model.safetensors', 'README.md')
MANIFEST = os.path.join(REPO_ROOT, 'p4mbfa', 'hf_release.yaml')
MANIFEST_MODEL_KEYS = ('suite', 'checkpoint', 'ckpt_name', 'trained_langs', 'train_data',
                       'selected_on', 'metrics', 'note')
# Keys only one suite takes. p3cupe checkpoints are published with the decoder they were
# re-scored with, not the one they were trained with, so their entries name it.
SUITE_KEYS = {'p4mbfa': (), 'p3cupe': ('decoder_overrides',)}
CODE_URL = 'https://github.com/tabahi/bfa_models'
G2P_URL = 'https://github.com/tabahi/CharsiuG2P'


def _standard_g2p():
    from local_paths import get_path
    sys.path.append(get_path('standard_g2p_repo'))
    from standard_g2p import gold_g2p as GP
    from standard_g2p import lang_group_inventory as LGI
    return GP, LGI


def build_release_config(ckpt, source_name, selection=None):
    '''config.json contents for a loaded training checkpoint. Raises on anything missing.

    `selection`: the manifest entry's trained_langs / train_data / selected_on / metrics / note,
    recorded under training.selection (None for a plain `export`).'''
    if 'model_config' not in ckpt or 'lang_group' not in ckpt['model_config']:
        raise ValueError(f"{source_name}: not a p4mbfa checkpoint (no p4mbfa model_config)")
    cfg = dict(ckpt['model_config'])
    hp = dict(ckpt['hyper_parameters'])
    missing = [k for k in (*GEOMETRY_KEYS, *DECODER_KEYS) if k not in hp]
    if missing:
        raise KeyError(f"{source_name}: hyper_parameters lack {missing}")

    GP, LGI = _standard_g2p()
    group = cfg['lang_group']
    tokens = list(LGI.tokens(group))
    phg_names = list(GP.phoneme_features.GROUP_NAMES)
    tone_vocab = list(GP.TONE_VOCAB)
    sizes = {'n_tokens': (cfg['n_tokens'], len(tokens)), 'n_phg': (cfg['n_phg'], len(phg_names)),
             'n_tones': (cfg['n_tones'], len(tone_vocab))}
    bad = {k: v for k, v in sizes.items() if v[0] != v[1]}
    if bad:
        raise ValueError(f"{source_name}: head sizes disagree with standard_g2p (checkpoint, "
                         f"tables): {bad}")
    if any(tokens[i] != t for t, i in SPECIAL_TOKENS.items()):
        raise ValueError(f"special tokens moved: {tokens[:4]}")

    exclude = list(hp.get('exclude_langs') or [])
    training = {
        **{k: hp.get(k) for k in PROVENANCE_KEYS},
        'epoch': ckpt.get('epoch'),
        'global_step': ckpt.get('global_step'),
        'source_checkpoint': os.path.basename(source_name),
    }
    if selection is not None:
        training['selection'] = dict(selection)
    return {
        'format': FORMAT,
        'format_version': FORMAT_VERSION,
        'model_config': cfg,
        'geometry': {k: hp[k] for k in GEOMETRY_KEYS},
        'decoder': {k: hp[k] for k in DECODER_KEYS},
        'labels': {
            'lang_group': group,
            'tokens': tokens,
            'special_tokens': SPECIAL_TOKENS,
            'phg_names': phg_names,
            'tone_vocab': tone_vocab,
            'member_langs': [l for l in LGI.member_langs[group] if l not in exclude],
            'gold_fingerprint': LGI.gold_fingerprint(),
        },
        'training': training,
    }


def model_weights(ckpt, source_name):
    '''The MbfaCNN tensors of a training checkpoint, without the `model.` prefix.'''
    sd = ckpt['state_dict']
    stray = [k for k in sd if not k.startswith('model.')]
    if stray:
        raise ValueError(f"{source_name}: tensors outside the model: {stray[:8]}")
    return {k[len('model.'):]: v.detach().cpu().contiguous() for k, v in sd.items()}


def build_model(config, state):
    '''MbfaCNN from a release config, with `state` loaded strictly.'''
    if config.get('format') != FORMAT or int(config.get('format_version', -1)) != FORMAT_VERSION:
        raise ValueError(f"not a {FORMAT} v{FORMAT_VERSION} config: "
                         f"{config.get('format')} v{config.get('format_version')}")
    model = MbfaCNN()
    model.load_config_state_dict(config['model_config'])
    model.make_model()
    model.load_state_dict(state, strict=True)
    return model


# ----------------------------------------------------------------------------- model card

def _metrics_table(metrics):
    rows = "\n".join(f"| `{k}` | {v} |" for k, v in metrics.items())
    return f"| metric | value |\n|---|---|\n{rows}\n"


def model_card(config, trained_langs=None, repo_id=None, ckpt_name=None):
    labels, train, cfg = config['labels'], config['training'], config['model_config']
    sel = train.get('selection')
    group = labels['lang_group']
    langs = list(trained_langs or labels['member_langs'])
    lang_note = ("the FLEURS languages it was trained on" if trained_langs else
                 "standard_g2p's members of the group")
    primary = sorted({l.split('-')[0] for l in langs})
    tone = ('trained on this group\'s tone layer' if group in
            ('cjk', 'thai_khmer', 'vietnamese') else
            'present but untrained (no language in this group has a tone layer)')
    front = "\n".join(["---", "license: agpl-3.0", "library_name: pytorch",
                       "language:", *[f"- {l}" for l in primary],
                       "datasets:", "- google/fleurs",
                       "tags:", "- forced-alignment", "- phoneme-recognition",
                       "- phoneme-segmentation", "- speech", "- multilingual", "---", ""])
    where = repo_id or "<this repo or directory>"
    title = f"{repo_id.split('/')[-1]} (p4mbfa · {group})" if repo_id else f"p4mbfa · {group}"
    hello = ['h', 'ɛ', 'l', 'o']
    example = hello if all(t in labels['tokens'] for t in hello) else labels['tokens'][4:8]
    example = json.dumps(['SIL', *example, 'SIL'], ensure_ascii=False)

    training = (f"Trained on FLEURS (`{train.get('data_src')}`), experiment "
                f"`{train.get('experiment')}`, epoch {train.get('epoch')}.")
    if sel:
        training = (f"**Data:** {sel['train_data']}.  \n**Checkpoint:** experiment "
                    f"`{train.get('experiment')}`, epoch {train.get('epoch')}, selected on "
                    f"{sel['selected_on']}.  \n{sel['note']}\n\n{_metrics_table(sel['metrics'])}")
        if any(k.startswith('val_') for k in sel['metrics']):
            training += ("\n`val_*` are on held-out FLEURS clips but are scored against the "
                         "model's own alignments (there are no boundary labels outside English), "
                         "so they measure self-consistency, not boundary accuracy.\n")
        if any(k.startswith('timit_') for k in sel['metrics']):
            training += ("\n`timit_*`: 400 TIMIT clips (English, never trained on). Each clip's "
                         "text is phonemised with standard_g2p, force-aligned, and the predicted "
                         "onsets are scored against TIMIT's hand-labelled phone boundaries: F1 "
                         "within 25 / 10 ms, and the mean absolute error of matched boundaries. "
                         "The clips are split in two halves: `tune` picked the checkpoint, "
                         "`holdout` did not. `nobonus` aligns without the log-mel onset bonus, "
                         "which shows the network's own share.\n")

    finetune = ""
    if ckpt_name:
        finetune = f'''
## Fine-tuning

`ckpt/{ckpt_name}` is the training checkpoint (PyTorch Lightning, weights only, a pickle: load
it only if you trust this repo). Download it, then point a p4mbfa yaml at it to continue
training, or to train a new language group on its trunk and shared `phg` head:

```bash
hf download {where} "ckpt/{ckpt_name}" --local-dir tmp/hf/{where.split('/')[-1]}
```

```yaml
ckpt_path: "tmp/hf/{where.split('/')[-1]}/ckpt/{ckpt_name}"
reset_fine_heads: true      # new lang_group: rebuild ph_head / tone_head; false = same group
```
'''

    return front + f'''
# {title}

A context-free CNN phoneme encoder and forced aligner (p4mbfa, the successor of CUPE / p3cupe),
for the **{group}** language group of [standard_g2p]({G2P_URL}). Each 5 ms frame is classified
from at most 120 ms of audio (38.9 ms receptive field), so the model cannot learn any language's
phonotactics. Phone boundaries come from a segmental Viterbi decoder over the known phone
sequence, refined to sub-frame precision at the crossing of neighbouring posteriors.

| head | labels | size |
|---|---|---|
| `ph` | local tokens of `{group}` (incl. `<blank> SIL noise <unk>`) | {cfg['n_tokens']} |
| `phg` | gold phoneme groups, shared by every language group | {cfg['n_phg']} |
| `tone` | tone values, shared by every language group; {tone} | {cfg['n_tones']} |

Languages ({lang_note}): {", ".join(langs)}. standard_g2p maps every member of the group
onto the same tokens, so other members may align too, untested.

## Training

{training}
Labels are standard_g2p dictionary pronunciations (gold inventory `{labels['gold_fingerprint']}`),
not phonetic transcriptions of what was said.

## Usage

The code is in [{CODE_URL}]({CODE_URL}) (`p4mbfa/`); clone it and run from its root.

```python
from p4mbfa.inference import MbfaAligner

aligner = MbfaAligner.from_pretrained("{where}")
wav = aligner.load_audio("clip.wav")                     # mono, {config['geometry']['sample_rate']} Hz
phones = {example}   # this group's tokens (config.json labels.tokens)
for seg in aligner.align(wav, phones):
    print(seg["token"], seg["start_ms"], seg["end_ms"])
```

`aligner.encode(wav)` returns the per-frame log posteriors of all three heads.
Text -> tokens is standard_g2p's job (`goldG2P.phonemize_sentence` then
`lang_group_inventory.to_local`); `config.json` lists the token strings. `from_pretrained`
downloads only `config.json` and `model.safetensors`.
{finetune}
License: AGPL-3.0.
'''


# ----------------------------------------------------------------------------- export

def export(ckpt_path, out_dir, trained_langs=None, selection=None, repo_id=None,
           ckpt_name=None):
    from safetensors.torch import save_file
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = build_release_config(ckpt, ckpt_path, selection)
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
          f"lang_group {config['labels']['lang_group']} ({config['model_config']['n_tokens']} "
          f"tokens); {card_note}")
    return config


# ----------------------------------------------------------------------------- manifest

def load_manifest(path, only=None):
    import yaml
    with open(path) as f:
        man = yaml.safe_load(f)
    for k in ('out_dir', 'namespace', 'models'):
        if k not in man:
            raise KeyError(f"{path}: missing top-level key '{k}'")
    models = man['models']
    for name, entry in models.items():
        if entry.get('suite') not in SUITE_KEYS:
            raise KeyError(f"{path}: model '{name}' needs `suite:` one of {list(SUITE_KEYS)}")
        keys = (*MANIFEST_MODEL_KEYS, *SUITE_KEYS[entry['suite']])
        missing = [k for k in keys if k not in entry]
        extra = [k for k in entry if k not in keys]
        if missing or extra:
            raise KeyError(f"{path}: model '{name}' missing {missing}, unknown {extra}")
    if only:
        unknown = [n for n in only if n not in models]
        if unknown:
            raise KeyError(f"--only {unknown}: not in {path} (have {list(models)})")
        models = {n: models[n] for n in only}
    out_dir = man['out_dir'] if os.path.isabs(man['out_dir']) \
        else os.path.join(REPO_ROOT, man['out_dir'])
    return out_dir, man['namespace'], models


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def release(manifest=MANIFEST, only=None):
    '''Build `<out_dir>/<name>/` for every manifest model: export + the renamed checkpoint.'''
    out_dir, namespace, models = load_manifest(manifest, only)
    for name, entry in models.items():
        src = _abs(entry['checkpoint'])
        if not os.path.isfile(src):
            raise FileNotFoundError(f"{name}: checkpoint {src} does not exist")
        folder = os.path.join(out_dir, name)
        selection = {k: entry[k] for k in ('trained_langs', 'train_data', 'selected_on',
                                           'metrics', 'note')}
        if entry['suite'] == 'p4mbfa':
            export(src, folder, trained_langs=entry['trained_langs'], selection=selection,
                   repo_id=f"{namespace}/{name}", ckpt_name=entry['ckpt_name'])
        else:
            from p3cupe.export_hf import export as export_p3cupe
            export_p3cupe(src, folder, entry['decoder_overrides'], entry['trained_langs'],
                          selection, f"{namespace}/{name}", entry['ckpt_name'])
        ckpt_dir = os.path.join(folder, 'ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)
        stale = [f for f in os.listdir(ckpt_dir) if f != entry['ckpt_name']]
        if stale:
            raise ValueError(f"{name}: {ckpt_dir} holds other files {stale}; it must hold only "
                             f"{entry['ckpt_name']} (remove them; they would be uploaded)")
        shutil.copy2(src, os.path.join(ckpt_dir, entry['ckpt_name']))
        print(f"   -> ckpt/{entry['ckpt_name']}")
    print(f"\n{len(models)} release folder(s) in {out_dir}. Check the model cards, then:\n"
          f"   python p4mbfa/export_hf.py push --dry-run")


def check_release_folder(folder, name, entry):
    '''Raise unless `folder` is exactly what `release` builds for this manifest entry.'''
    from safetensors.torch import load_file
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"{name}: {folder} does not exist; run "
                                f"`python p4mbfa/export_hf.py release` first")
    expected = {*RELEASE_FILES, os.path.join('ckpt', entry['ckpt_name'])}
    found = set()
    for root, _, files in os.walk(folder):
        for f in files:
            found.add(os.path.relpath(os.path.join(root, f), folder))
    missing, extra = expected - found, found - expected
    if missing or extra:
        problems = ([f"missing {sorted(missing)} (run `release`)"] if missing else []) + \
                   ([f"unexpected {sorted(extra)} (remove them)"] if extra else [])
        raise ValueError(f"{name}: {folder}: " + "; ".join(problems))
    with open(os.path.join(folder, 'config.json')) as f:
        config = json.load(f)
    src = config['training']['source_checkpoint']
    if src != os.path.basename(entry['checkpoint']):
        raise ValueError(f"{name}: config.json was exported from {src}, but the manifest now "
                         f"names {os.path.basename(entry['checkpoint'])}; rebuild with `release`")
    if config.get('format') != entry['suite']:
        raise ValueError(f"{name}: config.json is format {config.get('format')}, the manifest "
                         f"says suite {entry['suite']}; rebuild with `release`")
    if entry['suite'] == 'p4mbfa':
        build = build_model
    else:
        from p3cupe.export_hf import build_model as build
    build(config, load_file(os.path.join(folder, 'model.safetensors')))
    return sorted(found)


def push(manifest=MANIFEST, only=None, dry_run=False, private=False, message=None):
    '''Upload every release folder to `<namespace>/<name>`. Checks all of them before any upload.'''
    out_dir, namespace, models = load_manifest(manifest, only)
    plan = []
    for name, entry in models.items():
        folder = os.path.join(out_dir, name)
        files = check_release_folder(folder, name, entry)
        size = sum(os.path.getsize(os.path.join(folder, f)) for f in files)
        plan.append((name, folder, f"{namespace}/{name}"))
        print(f"{namespace}/{name}  <-  {folder}  ({size / 1e6:.1f} MB)")
        for f in files:
            print(f"     {f}")
    if dry_run:
        print("\n--dry-run: nothing uploaded.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    user = api.whoami()['name']
    print(f"\nLogged in to the Hub as {user}.")
    for name, folder, repo_id in plan:
        url = api.create_repo(repo_id, repo_type='model', private=private, exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo_id, repo_type='model', folder_path=folder,
            commit_message=message or f"Upload {name} ({models[name]['ckpt_name']})")
        print(f"pushed {repo_id}: {url} ({commit.oid[:10]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('export', help='one checkpoint -> one release directory')
    p.add_argument('checkpoint')
    p.add_argument('out_dir')
    p.add_argument('--trained-langs', default=None,
                   help='space-separated language codes for the model card (default: the '
                        "group's member languages)")

    for cmd, text in (('release', 'build every manifest model into out_dir/<name>/'),
                      ('push', 'upload the release folders to the Hub')):
        p = sub.add_parser(cmd, help=text)
        p.add_argument('--manifest', default=MANIFEST)
        p.add_argument('--only', nargs='+', default=None, metavar='NAME',
                       help='just these manifest models')
        if cmd == 'push':
            p.add_argument('--dry-run', action='store_true',
                           help='check the folders and list what would be uploaded')
            p.add_argument('--private', action='store_true',
                           help='create new repos as private (existing repos keep theirs)')
            p.add_argument('--message', default=None, help='commit message')

    args = ap.parse_args()
    if args.cmd == 'export':
        export(args.checkpoint, args.out_dir,
               args.trained_langs.split() if args.trained_langs else None)
    elif args.cmd == 'release':
        release(args.manifest, args.only)
    else:
        push(args.manifest, args.only, args.dry_run, args.private, args.message)
