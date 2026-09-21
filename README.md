# bfa_models

PyTorch models and training utilities used by
[bournemouth-forced-aligner (BFA)](https://github.com/tabahi/bournemouth-forced-aligner).
The current trainable model is CUPEv3 in [`p3cupe`](p3cupe/).

This README is a quick-start guide for training and fine-tuning. For the model and loss
details, see [`p3cupe/README.md`](p3cupe/README.md).

## Requirements

You need:

- Python with PyTorch, Lightning, PyYAML, and the other packages used by BFA.
- A CUDA-capable GPU. The training entry point is configured to use Lightning's GPU
	accelerator.
- A working checkout/install of BFA, which provides the phonemizer used to create training
	transcripts.
- Your own dataset and path lists. This repository does not include `speech_data_manager` or a
   dataset manager.
- Audio files with text transcripts that have been converted to phonemized `.ts` files.

Install BFA and its dependencies in the same environment as this repository. The exact install
command depends on where the BFA checkout is located; for a local checkout it is typically:

```bash
pip install -e /path/to/bournemouth-forced-aligner
```

## Configure Local Paths

From the repository root, create the machine-specific configuration:

```bash
cp .local_paths.example.yaml .local_paths.yaml
```

Edit `.local_paths.yaml` with paths that exist on your machine. The example names reflect the
original development setup, but `speech_data_manager` is not provided with this repository and
these paths are not required to point to that project:

```yaml
speech_data_manager_paths_manager: /path/to/speech_data_manager/paths_manager
metadata_base: /path/to/metadata_base
```

The file is ignored by Git. `speech_data_manager_paths_manager` is only an import path for the
module that provides your path lists; replace it with your own path-list manager. `metadata_base`
is a machine-specific root used to resolve relative metadata paths such as SRT and TS files. It
is not used to resolve audio paths. Add the TIMIT paths only if you use the optional TIMIT probe
or its data-preparation scripts:

```yaml
timit_paths_list: /path/to/paths_list_TIMIT_all.json
timit_test_paths_list: /path/to/paths_list_TIMIT_test.json
```

## Prepare Training Data And Path Lists

You are responsible for creating the dataset split and path lists. The training entry point
currently obtains those lists through the `splits.get_splits(data_src)` integration in
`p3cupe/train_cupe.py`; replace that integration with your own loader, or provide an equivalent
machine-specific path-list manager.

Each path-list entry must identify an audio file and its phonemized transcript, for example:

```python
{
   "audio": "/path/to/audio.flac",
   "ts": "/path/to/audio.ts",
}
```

Metadata paths may be absolute, or may use the `$/` prefix to resolve relative to `metadata_base`:

```python
{
   "audio": "/path/to/audio.flac",
   "ts": "$/transcripts/audio.ts.json",
}
```

The `.ts` file is JSON and must contain transcript segments with `ph` and `phg` arrays. A minimal
shape is:

```json
{
   "segments": [
      {"start": 0.0, "end": 1.2, "ph": [1, 22, 53], "phg": [1, 2, 4]}
   ]
}
```

The example path-list manager in [`data/paths_manager_example`](data/paths_manager_example/) shows
one way to map dataset names to JSON path-list files and split them for training. The exact split
representation is up to you, but the train, validation, and optional test lists must contain
entries in this format. `utils/data_prep_1.py` is a starting point for generating
the `.ts` files from BFA/Whisper-style transcript JSON. It is a library function, not a complete
dataset manager; adapt it or write your own phonemizer if your transcript format is different.

For the included helper, call `phonemize_texts` with a list of source/output files:

```python
from utils.data_prep_1 import phonemize_texts

phonemize_texts([
	{"lang": "en", "srt": "/path/to/input.json", "ts": "/path/to/output.ts"},
])
```

The utility does not overwrite existing `.ts` files unless called with `redo=True`.

## Run A Smoke Test

Before changing a model or configuration, run the synthetic end-to-end check:

```bash
python -m p3cupe.smoke_test
```

This does not need a dataset or GPU. It checks the model forward pass, backward pass, validation,
alignment, and output timestamp path.

## Train From Scratch

1. Copy the example configuration if you want a separate experiment:

   ```bash
   cp p3cupe/cupe_hp.yaml p3cupe/my_run.yaml
   ```

2. Edit `p3cupe/my_run.yaml`:

   - Set `experiment` to a unique run name.
   - Set `data_src` to the identifier your split/path-list loader expects, or replace the
   `data_src`-based split lookup in `p3cupe/train_cupe.py` with your own lists.
   - Set `gpu_num` to the GPU IDs to use.
   - Set `epochs`, `batch_size`, and `lr` for the run.
   - Leave `ckpt_path: null` for a new model.
   - For a random-weight start, use a nonzero `align_prior_epochs` during the early epochs.
   - Set `timit_probe_enabled: false` unless `timit_probe_cache` exists.

3. Run from the repository root. The command takes the YAML path **without** `.yaml`:

   ```bash
   python p3cupe/train_cupe.py p3cupe/my_run
   ```

For a short configuration/data check, set `dev_run: true` first. This uses Lightning's fast
development run and should be reset to `false` for a real training run.

## Fine-Tune Or Resume

Set `ckpt_path` to a saved checkpoint and give the run a new `experiment` name:

```yaml
experiment: "my_finetune"
ckpt_path: "/absolute/path/to/m_checkpoint.pt"
dev_run: false
```

Then run the same command:

```bash
python p3cupe/train_cupe.py p3cupe/my_finetune
```

The checkpoint supplies the model weights and saved architecture. The new YAML supplies the
dataset, optimizer, loss settings, validation settings, and schedule. Keep the architecture
compatible with the checkpoint; use `warm_start_arch_overrides` only for supported additive
changes.

## Checkpoints And Logs

Training outputs are written to:

```text
tmp/training/p3cupe/<experiment>/
```

The run saves the best monitored checkpoints and a final checkpoint named like:

```text
m_<experiment>_final_e<epoch>.pt
```

Metrics are logged to Weights & Biases using the `project` value in the YAML. Set up W&B for the
environment before a non-development run, or adjust the logger in `p3cupe/train_cupe.py` if your
workflow does not use W&B.

## Common Adjustments

- GPU out-of-memory: reduce `batch_size`, then consider reducing `duration_max` if long clips are
	not required.
- Slow runs: reduce validation/PER work with `light_val_metrics`, `per_enabled`, and the relevant
	`*_max_segments` settings.
- Fine-tuning instability: lower `lr`, change one setting at a time, and compare runs with the
	same decoder and boundary settings.
- Missing data or path errors: verify your local path-list integration, `data_src`, and the
   generated `.ts` transcript files.

## Useful Files

- [`p3cupe/cupe_hp.yaml`](p3cupe/cupe_hp.yaml): example training configuration.
- [`p3cupe/train_cupe.py`](p3cupe/train_cupe.py): training entry point.
- [`p3cupe/smoke_test.py`](p3cupe/smoke_test.py): dataset-free validation check.
- [`utils/data_prep_1.py`](utils/data_prep_1.py): transcript phonemization utility.
- [`p3cupe/README.md`](p3cupe/README.md): detailed model, loss, and hyperparameter reference.