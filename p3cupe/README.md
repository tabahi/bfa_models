# p3cupe
This is the 3rd major iteration of the CUPE model.

p3cupe learns frame-level phoneme posteriors from audio and uses those posteriors to place
phoneme boundaries. Training needs audio and phoneme sequences, but no boundary labels.

The main idea is:

1. Split each clip into short overlapping waveform windows.
2. Predict one phoneme distribution every 5 ms.
3. Align the known phoneme sequence to those predictions with a monotonic segmental decoder.
4. Turn that alignment into soft frame targets and train the model.
5. Read a boundary where neighboring phone posteriors cross.

The model is therefore both a phoneme recognizer and a boundary detector.

## Run It

```bash
python -m p3cupe.smoke_test
python p3cupe/train_cupe.py p3cupe/cupe_hp
```

Set `dev_run: true` before a real run. The YAML file is the source of truth for the new
training run, including a warm start. The checkpoint supplies model weights and the saved
model shape; the live YAML supplies training and loss settings.

## Data Flow

```text
waveform
  -> overlapping windows
  -> CNN feature extractor
  -> per-frame phoneme and group logits
  -> monotonic alignment against the known phone sequence
  -> frame targets
  -> cross-entropy loss
```

The default window is 120 ms and advances by 80 ms. Each window emits 24 frames, and the
overlap is stitched onto one continuous 5 ms grid. The CNN sees only local acoustic context;
it does not process a whole utterance at once.

## Model

The current trunk is:

```text
Conv 15 / stride 4
Conv  9 / stride 4       -> 80 samples per frame = 5 ms
Conv  7 / stride 5
channel gate
context convolutions      -> context_kernels
frame residual blocks     -> frame_blocks and ffn_mult
phoneme head + group head
```

The default `context_kernels: [5, 3]` gives a 38.9 ms convolutional receptive field. The
frame blocks add capacity without mixing time when `attn_radius: 0`. If attention is enabled,
each block can mix a local band of frames and the receptive field grows.

The output has `phoneme_classes + 1` phoneme logits and `phoneme_groups + 1` group logits.
The final extra slot is reserved for the shared label mapping and is not a training target.
There is no CTC blank class.

## Loss And Alignment

### Simple version

For every training clip, the phoneme sequence is known but its timing is not. The decoder tries
all legal monotonic segmentations and chooses the best one using the model's current frame
scores. Each phone must last at least `min_phone_frames`; optional extra duration is controlled
by `soft_phone_frames` and `soft_frame_cost`.

The chosen alignment becomes a soft, hat-shaped target. A phone is near 1.0 in its interior and
ramps toward 0.5 near its boundaries. The model is trained with cross-entropy to reproduce that
target. With `soft_target_mix > 0`, the target is instead mixed with a forward-backward average
over many legal alignments.

At inference, a boundary is the crossing of neighboring phone posteriors. `ramp_frames` controls
how wide that crossing is, and interpolation gives a sub-frame timestamp.

### Nerd version

The model produces `log p(c | x_t)`. The segmental dynamic program assigns frames to the known
phone chain while enforcing monotonic order and minimum duration. With `class_prior_norm`, the
alignment score uses

$$
s(c,t) = \log p(c \mid x_t) - \log \bar p(c),
$$

which reduces the incentive to explain every frame with a globally frequent class. The optional
diagonal prior is active only during a cold start and fades linearly over `align_prior_epochs`.

The Viterbi path produces the hard alignment. Forward-backward produces frame occupancy over
the same decoder topology. `soft_target_scale` controls how sharply that occupancy concentrates;
`soft_target_mix` blends occupancy targets with the hat target.

The onset bonus adds an acoustic-change score to transitions into phones. With
`boundary_flux_source: mel`, the score comes from L2 changes in log-mel features. This is useful
for acoustic changes, but it can prefer a burst inside a stop rather than the phone's true onset.
Use `boundary_bonus_weight: 0` when comparing ordinary phone-onset placement.

The total objective is phoneme cross-entropy plus `group_loss_weight` times the same kind of
cross-entropy for the coarse group head. The alignment is detached from gradient computation;
the network learns from the resulting targets.

## Hyperparameters

Change values in `cupe_hp.yaml`. The descriptions below give both the practical meaning and the
technical meaning.

### Run, data, and memory

| YAML key | Simple meaning | Technical meaning |
|---|---|---|
| `experiment` | Name of the output run | Output directory and checkpoint prefix |
| `project` | Weights-and-Biases project | Logger project name |
| `data_src` | Dataset to train on | Passed to the split manager |
| `gpu_num` | GPUs to use | Lightning device list |
| `epochs` | Number of epochs | Also sets the length of the OneCycle learning-rate schedule |
| `dev_run` | Tiny test run | Enables Lightning fast-dev-run behavior |
| `seed` | Repeatability seed | Seeds PyTorch initialization and sampling |
| `batch_size` | Clips per GPU batch | Dominant GPU memory control |
| `accumulate_grad_batches` | Steps to combine before updating | Effective batch is batch x accumulation x GPU count |
| `dataloader_n_workers` | Data loading workers | PyTorch DataLoader worker count |
| `fixed_window_batch` | Keep activation shapes constant | Computes fixed padded shapes; more predictable memory |
| `cuda_alloc_conf` | CUDA allocator setting | Applied before CUDA initializes |
| `duration_max` | Maximum clip length in seconds | Filters longer clips and sets the collator padding width |
| `ph_seq_max` | Maximum phone-sequence length | Limits collator and validation buffers |
| `val_max_segments` | Validation clips to score | Deterministic validation subset size |

Practical rule: lower `batch_size` first when memory is tight. Lower `duration_max` only when
the dataset really has shorter clips, because it filters data as well as changing padding.

### Window geometry

| YAML key | Simple meaning | Technical meaning |
|---|---|---|
| `window_size_ms` | Audio context seen at once | Waveform length given to the CNN |
| `window_stride_ms` | Distance between windows | Must advance by a whole number of 5 ms output frames |

Keep `window_stride_ms` compatible with the 5 ms frame grid. The default 120/80 geometry emits
24 frames and advances 16 frames, so stitching remains exact.

### Model capacity and context

| YAML key | Simple meaning | Technical meaning |
|---|---|---|
| `cnn_channels` | Width of the first CNN layer | Channels then grow to 2x and `embed_dim` |
| `embed_dim` | Width of each frame representation | CNN context width and classifier input width |
| `context_kernels` | How much nearby time the CNN mixes | Odd stride-1 kernels; each adds `(kernel - 1) * 5 ms` |
| `frame_blocks` | Number of residual processing blocks | Per-frame depth when attention is off |
| `ffn_mult` | Width inside each frame block | Hidden width is `ffn_mult * embed_dim` |
| `block_dropout` | Dropout in frame blocks | Regularization inside each block |
| `attn_radius` | Local attention reach | Each block attends `+/- radius` frames |
| `attn_heads` | Number of attention heads | Must divide `embed_dim`; used when radius is positive |
| `cnn_dropout` | CNN dropout | Dropout in convolutional blocks |
| `head_dropout` | Classifier dropout | Dropout before final phoneme logits |
| `noise_level` | Waveform noise augmentation | Training-time Gaussian noise scale |
| `phoneme_classes` | Number of real phoneme classes | Output size is this value plus one reserved slot |
| `phoneme_groups` | Number of coarse groups | Group output size is this value plus one reserved slot |

`context_kernels` and `attn_radius` change temporal context. `frame_blocks` and `ffn_mult`
change capacity. More context can blur boundaries; more per-frame capacity does not inherently
increase temporal context.

### Alignment and target shape

| YAML key | Simple meaning | Technical meaning |
|---|---|---|
| `min_phone_frames` | Minimum phone duration | Hard minimum in 5 ms frames |
| `soft_phone_frames` | Extra duration allowance | Optional decoder sub-states after the hard minimum |
| `soft_frame_cost` | Cost of skipping optional duration | Duration-prior cost per skipped optional frame |
| `sil_skippable` | Allow internal SIL to disappear | SIL states in the sequence may be skipped |
| `optional_sil` | Add optional SIL at clip edges | Absorbs leading or trailing silence |
| `boundary_bonus_weight` | Trust acoustic changes at onsets | Weight on the decoder's transition bonus |
| `boundary_contrast_frames` | Width of local flux contrast | Moving-average width used to make local peaks |
| `boundary_flux_source` | Source of onset change signal | `mel` uses fixed log-mel flux; `tap` uses learned CNN features |
| `ramp_frames` | Width of the boundary transition | Half-width of the hat target in frames |
| `group_loss_weight` | Strength of group supervision | Multiplier on group-head cross-entropy |
| `class_prior_norm` | Correct for globally common classes | Aligns using posterior divided by batch class prior; CE is unchanged |
| `soft_target_mix` | Use uncertainty from many alignments | 0 is Viterbi/hat; 1 is occupancy target |
| `soft_target_scale` | Make occupancy sharper or softer | Temperature-like scale for forward-backward path scores |
| `align_prior_epochs` | Gentle cold-start alignment | Diagonal prior fades to zero across this many epochs |
| `align_prior_sigma` | Width of cold-start diagonal band | Relative distance scale for the prior |
| `align_prior_strength` | Strength of cold-start pull | Nats charged for diagonal deviation |

For a warm start, keep `align_prior_epochs: 0` unless you deliberately want to pull a trained
alignment toward a flat split. If `min_dur_frac` is high, the minimum duration is binding; if it
is low, changing the duration floor is unlikely to help.

### Optimizer and checkpoint selection

| YAML key | Simple meaning | Technical meaning |
|---|---|---|
| `lr` | Maximum learning rate | `max_lr` for AdamW and OneCycleLR |
| `weight_decay` | Weight regularization | AdamW decoupled weight decay |
| `gradient_clip_val` | Limit large updates | Lightning gradient clipping value |
| `ckpt_monitor` | Metric used to keep checkpoints | ModelCheckpoint monitor |
| `ckpt_mode` | Better-is-higher or lower | ModelCheckpoint comparison mode |
| `early_stop_metric` | Metric used for stopping | EarlyStopping monitor |
| `early_stop_mode` | Direction for early stopping | EarlyStopping comparison mode |
| `patience` | Epochs without improvement | EarlyStopping patience |
| `ckpt_path` | Optional warm-start checkpoint | Loads weights and saved architecture |
| `warm_start_arch_overrides` | Explicit additive architecture changes | Allows only supported architecture growth during restore |

Warm-start rule: the live YAML controls loss, optimizer, data, validation, and schedules after
restore. The checkpoint controls tensor shapes so weights can load safely. Architecture changes
require the explicit `warm_start_arch_overrides` mechanism and must be additive.

### Validation and probe

| YAML key | Simple meaning | Technical meaning |
|---|---|---|
| `light_val_metrics` | Skip slow secondary metrics | Avoids Python-heavy validation diagnostics |
| `per_enabled` | Calculate PER/GER | Enables sequence error metrics |
| `per_mode` | PER decoding mode | Current supported mode is `greedy` |
| `per_max_segments` | Cap PER clips per GPU | Limits expensive edit-distance work |
| `per_every_n_epochs` | How often to calculate PER | Skips PER on other epochs |
| `timit_probe_enabled` | Run the labelled boundary probe | Validation-only external boundary check |
| `timit_probe_cache` | Probe data cache path | JSON cache consumed by the probe |
| `timit_probe_batch_size` | Probe batch size | Inference batch size for probe clips |
| `timit_probe_every_n_epochs` | Probe frequency | Runs the probe on matching epochs |

Useful readings are `val_loss`, `val_timit_nobonus_f1_25ms`, `val_timit_sub_f1_25ms`,
`val_frame_acc`, `train_min_dur_frac`, and `train_mean_phone_frames`. Model agreement with its
own alignment is not the same as correct boundaries, so do not use loss alone to choose a model.

## Safe Knob-Turning

1. Run the smoke test after code changes and use `dev_run: true` after config changes.
2. For warm starts, change one loss or optimizer knob at a time.
3. Compare checkpoints with the same decoder settings, especially the same
   `boundary_bonus_weight`.
4. If GPU memory fails, reduce `batch_size`; if training is slow, check `duration_max` and
   validation/PER settings.
5. For a cold start, use a nonzero `align_prior_epochs`. For a warm start, normally use zero.
6. Watch the boundary probe and duration statistics, not only the training loss.

## Files

| File | Purpose |
|---|---|
| `model.py` | CNN and frame-level output heads |
| `loss.py` | Segmental decoder, target construction, and loss |
| `trainer.py` | Windowing, stitching, training, and validation |
| `train_cupe.py` | YAML loading and Lightning entry point |
| `cupe_hp.yaml` | Active training configuration |
| `timit_probe.py` | Validation-only boundary probe |
| `datamodule.py`, `SegmentsCollator.py` | Dataset and batching |
| `windowing.py` | Window geometry and stitching |
| `metrics.py` | Validation metrics |