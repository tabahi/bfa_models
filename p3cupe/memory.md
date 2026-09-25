# p3cupe — decision record

`README.md` says **how it works**. This file is the **why, what was rejected, and what is still
unproven** — the parts that do not survive in code. Read both before changing the model or the
loss.

Pruned 2026-09-23: the run-by-run narrative of pa01–pa09 was removed once it stopped being
load-bearing (recoverable via `git show`). What survives is the reasoning, the measurements
still quoted elsewhere, and everything that must not be reintroduced. Numbers are TIMIT probe
holdout f1@25 unless said otherwise.

---

## START HERE — state as of 2026-09-24 (end of the pb02 session)

**The end goal: MFA-style boundary alignment from sequence labels only.** There are no boundary
labels in training. The objective builds its own targets by aligning the KNOWN phoneme sequence
to the audio using the model's own posteriors, so it is self-referential by construction and
its failure mode is drift.

### The answer so far, in five lines

1. **Best model: pa07f e0, decoded with min 6 + soft 10 + mel bonus 30 → f1@25 76.10 tune /
   75.58 holdout.** No training needed. For per-phone accuracy, decode the same model with
   bonus 0: 72.45 / 73.07 per-phone hit@25. MFA on the same clips: 79.1 f1@25.
2. **`cupe_hp.yaml` is now that recipe** (experiment `pc01a`): hard-EM, bonus 30 in the
   targets, min 6 + soft 10, ramp 3.0, libri_500h, 4 epochs, warm from pa07f, and every
   pb-era addition OFF (soft onsets, EMA, repr_flux, aug_views, local_delta). Dev run checked:
   probe 75.59 holdout / 76.03 tune at step 1, 37.1 GB peak. **Not yet trained.** Run:
   `/root/anaconda3/bin/python p3cupe/train_cupe.py p3cupe/cupe_hp`. It is judged against
   the step-1 numbers above; `ckpt_monitor` is now `val_timit_tune_f1_25ms`, so the holdout
   stays clean for reporting.
3. **The onset bonus in the TRAINING targets is the loop's anchor** and the only knob found
   that moves where training settles (bonus 10 / 3 / 0 end at no-bonus 72.2 / 70.6 / 68.1).
   pb01 lowered it from 30 to 10, and that alone explains the slide of every pb run. Keep
   decode bonus and training bonus as separate decisions.
4. **Everything else in the pb02 sweep is measured and dead** (table under "Results since the
   fix"): local_delta, ramp 3/4, repr_flux 0.3/3, lr 1e-5, aug_views, and offline snapping to
   any layer's change (`snap_sweep.py`). Nine arms at bonus 10 all end at 70.7–72.4. Do not
   re-run them.
5. **Untested and worth one run each, on the final recipe:** training-target bonus above 30
   (the trend was still rising at 10 → 30), and ramp 2.0 against 3.0 at bonus 30.

### Tools added this session
- `train_cupe.py p3cupe/cupe_hp key=value ...`: per-run overrides, values parsed as YAML;
  unknown keys raise. W&B records the resolved config.
- `run_pb02.sh [arm ...]`: sequential sweep queue. Waits for free GPUs before each arm; logs go
  to `tmp/training/p3cupe/logs/<arm>.log`. Never run arms in parallel: that changes the
  effective batch.
- `snap_sweep.py <ckpt>`: offline boundary snapping, also usable as a gate for any future
  boundary-signal idea. If it cannot beat the aligner here, a trained head on the same signal
  will not either.
- `decoder_sweep.py <ckpt> --half tune|holdout --bonus .. --min .. --soft ..`: re-decodes a
  checkpoint in minutes. **Run it on every new checkpoint before believing a training
  number.** The decoder settings were worth +2.2 on pa07f, more than any training change.
- Model: `local_delta` (off; zero-init classifier branch) and a printed WARNING when the yaml
  asks for an additive architecture key without listing it in `warm_start_arch_overrides`.
- Runs are deterministic: pb02f reproduced pb01b's e0–e3 to two decimals. Seed-to-seed
  variance is still unmeasured, so treat single-run differences under ~0.4 as noise.

### Gotchas hit this session
- PyYAML reads `1e-5` (no dot) as a STRING. The trainer casts `lr` with `float()`; anything
  new that reads a float from the yaml must do the same.
- The probe's frame-grid readout reports onsets at `o * 5 ms`. The hat's crossing is at
  `o − 0.5`, and that convention scores ~0.4 f1 lower against TIMIT. `dip_`/`flux_`/`sub_`
  carry that 2.5 ms handicap.
- An outside GPU job once emptied the whole queue in two minutes (every arm OOMed);
  `run_pb02.sh` now waits for free cards.
- `smoke_test.py` had a stale `sys.path` from the old `modules/` layout (fixed). The same
  stale line is still in `decoder_sweep.py`, `timit_probe.py` and others: run them with
  `PYTHONPATH=.` from the repo root.

**What to read, in this order:**

1. `val_target_entropy_ratio` — must sit at **1.000**. If it falls, the target has started
   chasing the model's own confidence and everything else is noise. This is the bug that caused
   the 2026-09-19..22 collapse.
2. `val_timit_tune_f1_25ms` (the monitor since 2026-09-24) and `val_timit_nobonus_f1_25ms`
   (the model's own signal): judge both against pa07f's step-1 numbers. The question is the
   **slope**, not the level.
3. `val_timit_dip_f1_25ms` — the falsification test: can the model segment with no transcript?
4. `val_timit_flux_f1_25ms` against the constant `bonusflux_f1_25ms` (60.34) — whether the
   trunk is keeping or discarding acoustic change.
5. `val_timit_onset_drift_frames` — decaying means the EM loop converged (possibly to a wrong
   place); flat or rising means it is still wandering.
6. `val_frame_acc` rising while the probe falls = self-agreement, not accuracy.

**Best surviving checkpoint:** `pa07f/m_pa07f_epoch=0_step=1246_...73.4052.ckpt` (75.6 holdout
f1@25 under the final decoder). pa01d e3, the historical best (75.8 headline, 75.4 per-phone),
was deleted in an earlier cleanup and is gone.

---

## The diagnosis that explains 2026-09-19..22

**`soft_target_mix: 1.0` was the drift.** `_soft_target` multiplies the hat by `1 - mix`, so at
1.0 the hat is not mixed down, it is **arithmetically deleted** — and with it `ramp_frames`, the
0.5 crossing, and the target's only fixed reference.

The hat's entropy is a function of the segmentation GEOMETRY, so it cannot move. The
occupancy's is a function of the MODEL'S CONFIDENCE, so it shrinks as the model sharpens:

    sharper posteriors -> sharper occupancy -> harder target -> sharper posteriors

Every self-referential number improved while the only externally-scored one collapsed:

| | pa07e e0 | pa08a e0 | pa09a e2 | pa09b e0 |
|---|---|---|---|---|
| `val_target_entropy` | 0.0859 | 0.0841 | 0.0704 | 0.0703 |
| `val_frame_acc` | 0.522 | 0.536 | 0.674 | 0.692 |
| `val_peak_kl` | 1.493 | 1.454 | 1.219 | 1.091 |
| `val_timit_nobonus_f1_25ms` | 72.66 | 72.23 | 67.77 | **63.55** |

### The fix — three switchable parts

1. **`soft_onset_mix`** puts the occupancy into the ONSET POSITIONS, not the class
   distribution. `occupancy_to_onsets` reads `E[onset_i] = sum_t sum_{s<i} gamma[t,s]` (the soft
   twin of `path_to_onsets`) and its variance off the same survival function; the hat is rebuilt
   around those fractional onsets via `onsets_to_phone_idx` + `peak_weights`. The hat and its
   fixed entropy survive, and the target gains sub-frame boundary positions an integer grid
   cannot carry. **Moments are float64 on purpose**: at T=2000, `E[X^2]` ~ 4e6 against an O(1)
   variance, and float32 returns a NEGATIVE variance, i.e. NaN sigma.
2. **EMA teacher** (`ema_decay`): the alignment is built from averaged weights, so a wrong
   boundary is not confirmed on the very next step. Held in a plain dict, **not** as a module
   attribute — `nn.Module.__setattr__` would register it, put it in the `state_dict` and break
   every older checkpoint. Pick the decay from the step count the run prints, not by feel:
   `limit_data` caps FILES while the dataset is SEGMENTS, so 10000 files is ~176 optimiser
   steps/epoch, not the ~23 the clip count suggests.
3. **`repr_flux_weight`**: keeps the embedding's frame-to-frame change correlated with log-mel
   change (Pearson per clip — shape only, bounded in [0, 2]). See "the wall" below.

`aug_views` / `aug_warp_range` (target onsets averaged over time-warped views, with the law of
total variance widening the ramp where views disagree) is implemented, **default off, never
measured**.

**Regression gate:** at `soft_onset_mix: 0, ema_decay: 0, repr_flux_weight: 0` the loss,
gradients and stats are **bit-identical** to the pre-change code across five configurations
(checked against `git show HEAD:p3cupe/loss.py`, including the old `soft_target_mix: 1.0`
path). Keep that property — it is what makes any A/B here readable.

---

## Results since the fix

**pb01a** (soft onsets + EMA teacher, ramp 3.0, `soft_onset_ramp_k: 1.0`, 10 epochs):

| ep | nobonus@25 | sub@25 | sub@10 | dip@25 | drift |
|---|---|---|---|---|---|
| 0 | 71.76 | 73.91 | 57.53 | 39.98 | – |
| 4 | 70.33 | 73.16 | 56.26 | 37.46 | 0.27 |
| 9 | 70.49 | 73.22 | 55.90 | 35.81 | 0.37 |

- **The runaway is gone.** `target_entropy_ratio` pinned at 1.000; `frame_acc` +0.01 over 10
  epochs (was +0.17); `peak_kl` −0.015 (was −0.40); `min_dur_frac` 0.104 → 0.107, no squeeze.
  Decline went from −4.2/epoch (pa09b) and −1.0/epoch (pa09a) to **−0.13/epoch, flat from e4**.
- **But it does not improve**, and it starts 1.6 below the warm start — switching away from a
  near-one-hot target costs that on the first epoch.
- **`soft_onset_ramp_k: 1.0` is harmful; set to 0.** `onset_sigma_frames` ~1.3, so ramps ran
  ~4.3 frames instead of 3, and both SHARPNESS readouts fell monotonically for it (dip 40.0 →
  35.8, sub f1@10 57.5 → 55.9) while the tolerant no-bonus f1 merely plateaued. This was the
  "keep the uncertainty" variant pa04a never tested. It is now tested: it buys nothing.
- **`onset_drift` does not decay** (0.21 e3 → 0.39 e8, p95 2.0) while every metric is flat: the
  loop wanders across segmentations that are equally self-consistent and equally mediocre.

**pb01b** (ramp 2.0, `soft_onset_ramp_k: 0`, `repr_flux_weight: 1.0`), epochs 0–3:

| ep | nobonus@25 | sub@25 | sub@10 | dip@25 | **flux@25** | drift |
|---|---|---|---|---|---|---|
| 0 | 72.95 | 74.85 | 58.90 | 40.02 | 60.14 | – |
| 1 | 72.44 | 74.51 | 58.08 | 38.48 | 62.26 | 0.50 |
| 2 | 72.18 | 73.94 | 57.37 | 35.76 | 62.25 | 0.37 |
| 3 | 72.25 | 74.10 | 57.66 | 37.04 | 62.76 | 0.33 |

- Narrowing the ramp and dropping the widening **recovered ~1.2 points of starting level**
  (72.95 against pb01a's 71.76) and holds a better level throughout.
- **The representation term does exactly what it was built to do.** `flux_f1@25` — the
  embedding's own transcript-free boundary signal — goes 60.14 → 62.76 and is for the first
  time in this lineage **above the parameter-free log-mel flux (60.34)**. In pb01a the same
  number FELL, 53.9 → 51.8. `repr_flux_corr` rises (user-observed at e4).
- **And none of it reaches the posteriors.** `dip@25` still 40.0 → 37.0, `nobonus` still
  drifting down ~0.2/epoch. See below.

**pb02a** (pb01b + `local_delta: 1`), 10 epochs, completed 2026-09-23:

| ep | nobonus@25 | frame-grid@25 | dip@25 | flux@25 | pb01b nobonus / dip |
|---|---|---|---|---|---|
| 0 | 72.97 | 74.72 | 40.34 | 60.17 | 72.95 / 40.02 |
| 1 | 72.34 | 74.41 | 38.64 | 62.31 | 72.44 / 38.48 |
| 2 | 72.23 | 73.89 | 35.22 | 62.35 | 72.18 / 35.76 |
| 3 | 72.30 | 73.83 | 37.71 | 62.45 | 72.25 / 37.04 |
| 6 | 72.20 | 73.57 | 33.59 | 62.77 | – |
| 9 | 71.79 | 73.38 | 34.49 | 62.16 | – |

- **The head USED the branch and it changed nothing.** `train_delta_share` ended at 0.18
  (the delta branch is 18% of the hidden pre-activation), yet epochs 0–3 match pb01b to within
  noise on every readout. The same slow decline continues to e9 (nobonus −1.2, dip −5.9).
- This **falsifies "the head cannot see the change"** as the reason the dip does not follow
  flux. Given the change directly, the classifier spends it on something the hat target
  rewards, not on a dip. So the bottleneck is the OBJECTIVE, not what the head can see: the hat
  asks for flat posteriors in the phone interior and a crossing ±2 frames wide, and a sharper
  crossing gives a NARROWER dip, not a deeper one. The dip readout may be the wrong instrument
  for a model that fits the hat better. Read frame-grid and sub f1 before treating dip as
  ground truth. They fell too (74.7 → 73.4), so this is not only an instrument effect.
- End state: `target_entropy_ratio` 1.000, `repr_flux_corr` 0.88, `frame_acc` 0.60,
  `onset_drift` 0.43 (p95 2), `min_dur_frac` 0.108. No runaway and no squeeze.
- Verdict: leave `local_delta` at 0 unless a later objective change gives the head a reason
  to use it. The other pb02 arms first ran into an outside 42 GB GPU job and OOMed within
  20 s. They were relaunched at 14:59; `run_pb02.sh` now waits for free cards.

**pb02 sweep** (all warm from pa07f, 10 epochs, one knob off pb01b; e0 → e9, TIMIT holdout f1@25):

| arm | change | nobonus | frame-grid | dip | flux | peak_ce |
|---|---|---|---|---|---|---|
| pb02f | control | 72.95 → 72.23 | 74.70 → 73.84 | 40.0 → 33.1 | 60.1 → 62.5 | 1.372 |
| pb02a | `local_delta: 1` | 72.97 → 71.79 | 74.72 → 73.38 | 40.3 → 34.5 | 60.2 → 62.2 | 1.316 |
| pb02b | ramp 3.0 | 72.63 → 71.42 | 74.45 → 73.27 | 40.8 → 34.6 | 60.0 → 62.3 | 1.419 |
| pb02e | ramp 4.0 | 72.14 → 70.73 | 74.22 → 72.94 | 40.9 → 35.4 | 59.5 → 62.2 | 1.468 |
| pb02c | repr 3.0 | **73.32** → **72.43** | **75.01** → 73.78 | 40.0 → 33.9 | 62.7 → 62.5 (peak 63.3) | 1.386 |
| pb02d | repr 0.3 | 72.59 → 72.31 | 74.61 → 73.57 | 39.5 → 33.7 | 57.8 → 62.1 | 1.364 |

- **No arm improves on the warm start.** pa07f e0 is 73.41; the best any arm reached is 73.32
  (pb02c e0). Every arm declines after epoch 0, whatever the knob. The decline belongs to the
  training loop itself, not to any setting tested.
- **The receptive-field-floor hypothesis is refuted.** Wider ramps are monotonically WORSE on
  every tolerant readout (e0 nobonus: ramp 2.0 72.95 > 3.0 72.63 > 4.0 72.14, same at e9).
  Only the dip improves slightly with width, and that is the instrument: a wider hat means a
  wider dip. If anything, the trend points NARROWER (1.5) - untested.
- **Dip does not follow flux, at any weight.** Final dip sits at 33–35 in every arm, while final
  flux spans 62.1–62.5 and peak flux 62.1–63.3. At repr 0.3 flux starts BELOW log-mel (57.8)
  and still climbs to 62.1, so part of that rise comes from the loop, not the term.
- **repr 3.0 is the only weak positive:** +0.37 at e0, +0.20 at e9 over control, with no
  identity cost visible (peak_ce 1.386 against 1.372). Within what one seed can resolve? Unknown -
  no seed replicate exists.
- All arms were healthy: `target_entropy_ratio` 1.000, `min_dur_frac` 0.108–0.110,
  `onset_drift` 0.34–0.43.

**pb02g** (control + `lr: 1e-5`, 5× lower), 10 epochs, 2026-09-24:

| ep | 0 | 1 | 2 | 3 | 5 | 7 | 9 |
|---|---|---|---|---|---|---|---|
| nobonus, lr 1e-5 | 73.11 | 73.00 | 72.72 | 72.41 | 72.15 | 72.34 | 71.98 |
| nobonus, lr 5e-5 (pb02f) | 72.95 | 72.44 | 72.18 | 72.25 | 72.25 | 71.92 | 72.23 |
| dip, lr 1e-5 | 39.0 | 40.1 | 39.9 | 39.8 | 37.0 | 36.1 | 35.6 |

- **The decline is NOT proportional to lr; the loop has an ATTRACTOR.** With a 5× smaller step
  budget the run leaves the warm start more slowly (dip holds ~40 to e3, onset_drift 0.19
  against ~0.37) but ends at the same place: nobonus 71.98 against 72.23, frame-grid 73.75
  against 73.84, dip 35.6 against 33.1. Even the first low-lr epoch loses 0.3 (73.41 → 73.11).
- Across all seven pb02 arms, the e9 no-bonus lands in 70.7–72.4. The knobs move the level of
  the fixed point a little (ramp width moves it most); nothing tested moves it above the
  warm start. **The current objective's equilibrium sits BELOW pa07f**, so lr, EMA, repr and
  head changes can only change how fast it gets there. Improving the score means changing
  what the loop converges TO, i.e. a signal from outside the loop.
- Healthy throughout: `target_entropy_ratio` 1.000, `min_dur_frac` 0.110, `peak_ce` 1.402.
- Practical corollary until then: the best model this objective produces is its warm start
  plus at most one gentle epoch. Treat further training as a way to test objectives, not to
  improve checkpoints.

**pb02h** (control + `aug_views: 2, aug_warp_range: 0.05`, `soft_onset_ramp_k` 0, so the
views only AVERAGE the target onsets), 10 epochs, 2026-09-24. 4 h 20 min, 2.1× the control:

| ep | 0 | 2 | 4 | 6 | 9 |
|---|---|---|---|---|---|
| nobonus, aug 2 | 72.87 | 72.31 | 71.92 | 72.20 | 72.25 |
| nobonus, control | 72.95 | 72.18 | 72.08 | 72.12 | 72.23 |
| dip, aug 2 | 39.8 | 34.9 | 33.5 | 34.3 | 33.7 |

- **No effect at all:** the same curve and the same fixed point, 72.25 against 72.23.
  `onset_drift` 0.39, i.e. the loop wanders as much as it does without the views.
  `onset_sigma_frames` 1.93, so the views DO disagree by about 2 frames, but averaging removes
  only that symmetric scatter, and the scatter is not what sets the fixed point.
- Reading: the loop's bias commutes with a ±5% warp. That fits a bias that is a constant
  FRACTION of a phone (invisible to warping by construction, see `_warped_onsets`). It does
  not fit a constant-frame bias such as the burst pull, which the warp was aimed at. The one
  implemented outside-the-loop signal is therefore measured and spent. Leave `aug_views` at 0:
  it costs 2× and buys nothing.

**pb02i / pb02j** (control with the TRAINING-target onset bonus 3.0 / 0.0), 2026-09-24:

| bonus in targets | e0 | e3 | e6 | e9 | frame_acc | min_dur_frac |
|---|---|---|---|---|---|---|
| 10 (pb02f) | 72.95 | 72.25 | 72.12 | 72.23 | 0.579 | 0.109 |
| 3 (pb02i) | 72.68 | 71.44 | 70.80 | 70.64 | 0.601 | 0.115 |
| 0 (pb02j) | 72.41 | 69.57 | 68.93 | 68.07 | 0.611 | 0.118 |

- **The bonus IS the anchor, and the fixed point moves monotonically with it.** Lower bonus
  gives faster, deeper self-agreement drift (frame_acc up, probe down, short phones squeezing).
  This reproduces pa03a, now WITH the EMA teacher: the teacher does not replace the bonus.
- **pa07f was trained at bonus 30** (checked in its hyper_parameters). Every pb run warm-starts
  from it at 10. This is the most likely explanation for "every arm declines from the warm
  start" (pb02g): pb01 lowered the anchor, which lowered the fixed point, and each run then
  slides to it. The EMA / repr / lr / warp arms could not change it because none of them touch
  the anchor.
- **Separate the two uses of the bonus.** At DECODE it is monotonically harmful per phone
  (decoder sweep, 30 → 0 gives 68.0 → 74.6, the burst pull). In the TARGETS it is what holds the
  loop off a self-consistent wrong segmentation. The yaml moved both down together, but only
  the decode side should move: train with a high bonus, and read/decode with bonus 0 (which is
  what `nobonus_` already reports).
- Untested next arm: bonus 30 in the targets (pa07f's own setting) - does the no-bonus
  readout then hold at 73.4 or rise, rather than slide? Bonus 30 was the setting before the
  soft_target_mix fix; the pb-era fixes (soft onsets, EMA, correct OneCycle) have never been
  combined with it.

---

## Where the wall is now — the sharpest statement we have

Transcript-free boundary f1@25, same 200 clips, same scorer:

| signal | pb01a | pb01b |
|---|---|---|
| log-mel flux, parameter-free (`bonusflux_`) | 60.34 | 60.34 |
| the model's embedding, frame-to-frame change (`flux_`) | 53.9 → 51.8 | **60.1 → 62.8** |
| the model's own posterior confidence dip (`dip_`) | 40.0 → 35.8 | 40.0 → 37.0 |
| random-init floor | 29.9 | 29.9 |

Before `repr_flux_weight` the trunk was discarding boundary information — the identity
objective rewards invariance to exactly the acoustic change that marks a boundary. That is now
fixed, and measurably so. **The information is in the embedding and does not survive the
classifier.**

Two reasons this is probably not a weighting problem:

- **`dip_abs_err` is 29–30 ms and the receptive field is 38.9 ms.** The dip's precision is
  pinned near the RF. A `ramp_frames: 2.0` target asks for a ±10 ms transition, sharper than
  38.9 ms of context can resolve — the target may now be asking for something the architecture
  cannot produce, whatever the embedding carries.
- The hat target explicitly asks the classifier for `w = 1.0` across the whole phone interior,
  i.e. to be **invariant to within-phone change**. That is correct for identity, and it is
  exactly what suppresses the dip everywhere except a 2-frame window.

**Reasoned next steps (as of 2026-09-23) - ALL THREE TESTED in pb02, none moved the dip; see the pb02 sweep above:**

1. Sweep `ramp_frames` back up (3.0, 4.0) now that `repr_flux` is on — the RF argument says 2.0
   may be under the resolution floor. Config only.
2. Sweep `repr_flux_weight` (0.3 / 3.0) to see whether `dip` ever follows `flux`.
3. Give the classifier an explicit local-change feature — concatenate a short-range embedding
   difference onto the classifier input — so the head can place a dip without needing the trunk
   to move. **Implemented as `local_delta`** (`model.CupeCNN.classify`): both one-sided
   differences `[e_t - e_{t-1}, e_{t+1} - e_t]` through a zero-initialised `delta_proj` added
   to the classifier's hidden layer (group head untouched). Correction to the note above: it
   DOES widen context, for the classifier only - one frame each way, 38.9 -> 48.9 ms. The
   embedding, the repr term and the self-test's 38.9 ms are unchanged. Warm start from pa07f is
   bit-identical (max |diff| 0.0, checked) via `warm_start_arch_overrides: [local_delta]`;
   without that line the checkpoint's shape wins and a WARNING prints. +1.05M params, +2 GB
   peak at batch 108. Read `train_delta_share` (branch norm / embedding-branch norm at the
   hidden layer): near 0 means the head never used it and the run says nothing.

Standing caveat: pa04a measured that **66% of far misses have a posterior onset spread under
10 ms — the model is CONFIDENTLY wrong**. If that still holds, nothing that reweights the
model's own beliefs can move those boundaries.

---

## Ideas discussed, not started

**Boundary spike head (user, 2026-09-23).** A parallel head on the CNN embedding that emits
isolated 1s: the count equals the number of phones, every 1 has 0s on both sides, and the
spacing comes from matching the classifier's posterior crossings within a leeway radius. The
motivation: the older CUPE snapped boundaries to sudden embedding change (layer and frame radius
as knobs), and that worked better on CNN embeddings than on MLP logits. This repo's numbers agree:
embedding flux f1@25 is 62.8 against a posterior dip of 37.

Discussion points, before anyone builds it:
- Without identity, "N isolated spikes" is CTC with one repeated label, so it has the same
  unconstrained timing: any N positions score equally well. The drift toward the start of the
  sentence follows from that. As a soft loss (sigmoid, expected count = N) it also has a
  smeared minimum at p = N/T that barely pays the isolation penalty.
- The spacing term is what carries the information. Matched to the classifier's crossings, it
  distils the aligner and cannot beat it. This is the rejected `BoundaryHead` again: supervised
  by Viterbi onsets it scored 54.4 against raw flux at 59.5.
- The forced aligner already enforces count = N, monotonic spacing and a minimum duration, and
  it also knows the phone identities. The only new thing the head adds is where it reads from:
  the local CNN layer. `local_delta` (pb02a) tests that part at a fraction of the cost.
- **Cheapest test, no training:** snap the Viterbi onsets to embedding-flux peaks within radius
  r (sweep the tap × r ∈ {1, 2, 3}) and score no-bonus f1@25 and per-phone. If snapping does
  not help, a head trained toward the same signal will not help either. Caveat: the mel-flux
  onset bonus is monotonically worse per phone, because it drags onsets onto stop bursts.
- **TESTED 2026-09-24 (`snap_sweep.py`), result: no effect.** 108 configs per checkpoint: taps
  fe/ctx/emb/hid/post/mel × span 1,2 × radius 1–3 × peak-z gate −∞/0/1, on pa07f e0, pb02f
  final and pb02c final. The no-bonus aligner does the placing; snapping only moves each onset.
  The config is picked on the tune half and read on the holdout half:

  | ckpt | baseline hold f1 / phone | tune pick | pick's hold f1 / phone |
  |---|---|---|---|
  | pa07f e0 | 73.41 / 72.79 | fe k1 r2 z1 | 73.27 / 72.66 |
  | pb02c e10 | 72.43 / 72.13 | fe k1 r2 z1 | 72.39 / 72.00 |

  (with `--offset 0.5`, see below.) The tune gain (+0.1 to +0.5 per phone) does not transfer,
  and every holdout difference is within ±0.3, i.e. noise. **The change signal in no layer
  carries boundary timing that the forced aligner lacks.** The layer ORDER does reproduce the
  old CUPE finding: the earliest CNN tap `fe` is consistently the best snap target, and
  emb/hid/post/mel are all tied and worse. That only means fe does the least harm. Per class,
  fe snapping helps vowels (+0.4–0.9) and hurts nasals (−1.3 to −1.6).
- Side finding, conventions: reporting onsets at `o − 0.5` frames (where the hat puts the 0.5
  crossing) scores ~0.4 f1 and 0.6–1.0 per-phone BELOW reporting them at `o`, on every
  checkpoint. The aligner's onsets land about half a frame early against TIMIT's labels (or
  its convention), so the probe's `o * ms` readout is the better-calibrated one. Any readout on
  the `i − 0.5` convention (dip_, flux_, sub_) carries that 2.5 ms handicap.
- Verdict on the spike head: its spacing term would have to supply timing that no layer
  carries, so do not build it. Re-propose only if some new signal passes this snap test first.

---

## The objective — what was chosen and why

- **Hat-shaped target, no blank.** Each phone's posterior peaks at 1.0 across its interior and
  falls linearly to **exactly 0.5 at each boundary**, where the neighbour rises through 0.5. The
  boundary is the **crossing point of two posteriors** — continuous, recoverable by
  interpolation (`refine_onsets`), which is why 5 ms frames are not the resolution limit.
- **Why not CTC.** CTC marks a phone with a spike and fills the gaps with blank; *where inside a
  blank run the boundary lies is not represented in the output at all*. p2cupe measured
  placement inside a blank run at chance (26.8 ms observed against 25.9 ms predicted by random
  placement in a 4.85-frame run). Snapping onto spectral flux, a learned boundary head and a
  partition CE were all attempts to recover information the loss had already discarded, and all
  three measured out at ~zero or negative.
- **Targets come from the model itself**: monotonic Viterbi against the known phoneme sequence,
  under `no_grad` — classic hard-EM, the bootstrap HTK/MFA use. Minimum phone duration is
  enforced by expanding each phone into `min_phone_frames` sub-states.
- **Flat start = the diagonal band prior**, weight 1 → 0 over `align_prior_epochs`. At weight 1
  the alignment is "divide the clip evenly between the phones". Set to 0 when warm starting.
- **Optional SIL at both ends** (MFA's optional silence). Without a blank nothing absorbs
  leading silence, and TIMIT clips open with ~130 ms of it.
- **The target has entropy**, so the CE cannot reach 0. `peak_kl` (CE minus target entropy) is
  the number that goes to zero. **This is the property `soft_target_mix: 1.0` destroyed.**
- **The group head is the same loss at coarser granularity**, at a fixed `group_loss_weight`.
  Not a second objective: no schedule, no ramp, no priority controller.

**A known bias in the hat, currently masked.** `ramp` is clipped to half the segment, so a
phone's total `sum log w` saturates at −0.914 for `L >= 2*ramp` but is *cheaper* below it
(−0.604 at L=4, −0.288 at L=2). A perfectly fitted model therefore scores a **higher** Viterbi
path by squeezing phones onto the duration floor — the `min_dur_frac` climb seen whenever the
duration prior is relaxed. `soft_frame_cost` currently over-corrects it (a 3-frame phone with
`soft_phone_frames: 10` pays 10 nats against the hat's ~0.6), so it is not the active problem,
but widening ramps enlarges the biased region. Watch `min_dur_frac`.

## The model — what was chosen and why

- **CNN only, no transformer.** p2cupe's per-layer probe scored boundary placement at every tap
  and the ordering was stable: layers that *cannot* mix across time scored ~20 f1 points above
  layers that can. A module that smears 150 ms into one frame cannot say where a 10 ms event
  starts, and no loss downstream of it can undo that.
- **5 ms frames** (conv strides 4·4·5 = 80 samples). 13 ms put a ~6.5 ms quantisation floor
  under every boundary before the model did anything.
- **38.9 ms receptive field**, from two stride-1 context convs (`context_kernels: [5, 3]`).
  Widening walks back toward the time-mixer that destroyed p2cupe's boundaries.
- **Every conv has an odd kernel with `padding = kernel // 2`**, which is what puts frame `f` on
  input sample `f * 80` with no offset term anywhere. An even kernel shifts every timestamp by
  half a frame and nothing raises — `model.py`'s self-test asserts this with a click at a known
  sample. **Keep that test.**
- **Windowing kept** (120 ms / 80 ms stride): removing context is what makes the model
  multilingual, and 120 ms is too short to memorise any language's phonotactics. The stride
  advances exactly 16 of 24 frames, so the stitched grid *is* the conv grid and
  `time_ms = frame * 5.0` is exact.

---

## Do not reintroduce — measured and rejected

| idea | what happened |
|---|---|
| `soft_target_mix: 1.0` (occupancy as the class distribution) | deletes the hat; the entropy runaway above. Use `soft_onset_mix`. |
| `soft_onset_ramp_k > 0` (widen ramp by onset sigma) | pb01a: dip 40.0 → 35.8, sub f1@10 57.5 → 55.9. Buys nothing. |
| soft-EM in general | pa04a: per phone 74.1 holdout against a 75.4 start. The misses are CONFIDENT — 66% of far misses have spread < 10 ms. |
| transformer | the worst layer in its own boundary probe |
| CTC + blank + blank/SIL penalties | the boundary is not representable in its output |
| `BoundaryHead` + BCE | supervised by Viterbi onsets, i.e. distilled from a *worse* teacher than raw flux (54.4 against 59.5) |
| flux hinge (`bnd_lambda`) | inert at low weight, destructive at high (66.2 → 40.4 f1@25 in two epochs) |
| partition CE + lambda ramp | mechanism worked perfectly (`direct_transition_frac` 0 → 0.98) and moved the metric not at all, three rounds running |
| local attention for identity (pa02a) | +1 point frame acc, lower CE, boundaries unmoved; every class row within ±1.5. 35% slower. |
| wider windows / wider per-frame context (pa06) | 360 ms + attention was not better than 120 ms; re-windowing alone cost ~2 points |
| removing the onset bonus from the TARGETS (pa03a) | declined every epoch; the bonus was the only non-self-referential anchor |
| stronger duration prior (pa03b/c) | stops the squeeze, boundaries still below the start. The squeeze was a symptom, not the cause. |
| per-class / obstruent-only bonus scaling | effects are not additive per class; remove the pull on a vowel and the same flux peak drags the neighbouring consonant onto it |
| sub-phone (multi-state) phones, MFA-style (pa05) | gave the first per-phone gain over pa01d (76.2 against 75.4) but drifted again when trained at the better topology, and nasals fell 84 → 76. **Rolled back by the user; `states.py` deleted.** Re-propose only with new evidence. |
| CTC anneal, dual-loss priority controller, curriculum imbalance | schedules tuning schedules |

**The onset bonus itself.** The decoder sweep is unambiguous that per phone it is monotonically
worse (30 → 10 → 3 → 1 → 0 gives 68.0 → 70.1 → 71.6 → 73.4 → 74.6) because it drags onsets onto
stop bursts. The `levenshtein` headline moves the opposite way only because it credits TIMIT's
extra closure→burst boundaries — it rewards the bonus for the wrong reason. It is now at 10,
down from 30, with the EMA teacher as the anchor instead.

---

## Evaluation

- **`metrics.py` is a verbatim copy of p2cupe's and must stay that way** — it is the unchanged
  yardstick. Only how much is fed to it may change.
- **PER/GER are guards, not targets.** In p2cupe they were *anti*-correlated with placement, so
  `ckpt_monitor` is a boundary f1. Currently `per_enabled: false` and `frame_acc` is the
  identity guard — remember it measures agreement with the model's OWN alignment, not accuracy.
- **The TIMIT probe is the real instrument** (validation only, never trained on), with readouts
  `(none)` frame-grid, `sub_` (interpolated crossing), `dip_` (free peak-pick on the confidence
  dip, **no transcript** — the falsification test), `flux_` (embedding change), `bonusflux_`
  (the fixed log-mel flux the bonus reads), and `nobonus_` (same aligner, bonus zeroed — the
  monitor). Odd clips are the holdout (`val_timit_*`), even are the tuning half (`tune_*`).
- **Measured floors at random init** (400 clips): decoder 22.5, sub 22.7, dip 29.9, flux 37.1
  f1@25. A plain uniform split scores 52.4. **Read the gaps, not the absolutes.**
- **Reference points.** MFA on the same 400 clips: 79.1 f1@25 on the standard scorer, 83.7
  per-phone hit@25. pa01d e3 was 75.8 headline / 75.4 per-phone decoded with bonus 0.
- **The per-phone score is the honest one.** `miss_breakdown.py` matches phone sequences instead
  of boundaries, so TIMIT's extra closure→burst boundaries cannot inflate it. Run
  `decoder_sweep.py` / `miss_breakdown.py` / `error_breakdown.py` on the best checkpoint, with
  bonus 0, before believing any headline.
- **Where the remaining gap to MFA lives**: gradual transitions. Vowel onsets 67 against 86,
  lateral 57/77, rhotic→vowel 54/81, glide→vowel 51/78. Stops are a *convention* difference,
  not an error — p3 starts a stop at its BURST, MFA and TIMIT at its CLOSURE.
- **f1@10 against MFA, before using it as a selling point**: p3 leads 58.9 to 53.6 on the
  standard scorer, but scored on phone onsets only MFA leads at every tolerance (57.7 to 52.1).
  The claim holds on the benchmark; it rests on the burst convention.

---

## Traps already hit — do not reintroduce

- **Sequences are padded with `-100`** by `SegmentsCollator`. The DP gathers emissions for every
  column, so a negative id is an out-of-bounds index → a CUDA device-side assert several kernels
  later, pointing somewhere else. Clamped once in `prepare_targets`.
- **The padding shortcut must include the window that overlaps the tail**
  (`(max_frames-1) // stride_frames + 1`, not "enough windows to reach it"). One short changes
  the value of the last ~8 frames of every clip, silently. `smoke_test.py` asserts this.
- **p2cupe's TIMIT probe fed audio at 1/10 the training level.** BatchNorm hides most of it.
  p3's probe matches the collator exactly.
- **Stitching accumulates in float32** even under bf16 autocast: the boundary readout
  interpolates the *difference* between adjacent frames' logits, and bf16's ~0.06 resolution at
  logit 10 lands directly on that quantity. The same applies to the EMA teacher's forward.
- **`occupancy_to_onsets` needs float64** — float32 returns a negative variance, i.e. NaN sigma.
- **`searchsorted` does not check sortedness**, it returns nonsense. Any blended onset tensor
  must be `cummax`-ed first.
- **The EMA teacher must not be a module attribute** — `nn.Module.__setattr__` registers it and
  every older checkpoint stops loading.
- **`dev_run: true` writes `m_<exp>_final_e1.pt` under the same experiment name**, so never
  warm-start from that file by accident and never wait on it to detect the end of a real run.
  Wait for the process to exit.
- **`last.ckpt` is the last MONITORED checkpoint**, not the final epoch; the final weights are
  `m_<exp>_final_e<N>.pt`.

### Fixed 2026-09-23: the OneCycle schedule never annealed

`configure_optimizers` passed `len(train_dataloader())` — *batches* — as `steps_per_epoch`, but
Lightning steps an `interval: 'step'` scheduler once per **optimiser** step. With
`accumulate_grad_batches: 2` the cycle was twice as long as the run: pb01a configured 1760 steps
and took 880, so the lr peaked at epoch 6 and ended at ~4e-5, never annealing. **Every run
before pb01b has this bug** — read their late-epoch numbers with it in mind; pb01a's metric dip
at e6–e8 and its recovery at e9 track the lr peak, not the loss.

Related and still true: `epochs` is a learning-rate knob. OneCycle spreads warmup over
`pct_start=0.3` of the whole run, so at `epochs: 100` the first ~10 epochs run at a tenth of
`lr` and look dead. p2cupe lost 10 GPU-hours to this.

---

## The risk to keep in mind

The targets come from the model's own alignment, so **the loss can be driven to zero by a
segmentation that is perfectly self-consistent and systematically wrong** — every boundary 20 ms
early, say. Nothing inside the objective can detect that; it is the known failure mode of
hard-EM and the price of having no boundary labels. The guards are all external: the diagonal
prior while the model is random, the TIMIT probe every epoch (labels training never sees),
`train_mean_phone_frames` against the corpus's real phone rate, and now `target_entropy_ratio`
and `onset_drift`.

**Watch the probe, not the loss.**

---

## Engineering notes

**Why two DDP ranks report different GPU memory.** Not different work. The peak is set by the
longest clip in a batch when `fixed_window_batch: false`, and the caching allocator never
returns memory, so each rank strands a different amount. Measured over 30 steps, 1–10 s clips:
ragged shapes 4.03–4.46 GB (0.43 swing), ragged + `expandable_segments` 3.10–3.14 (0.04), fixed
shapes 3.95 (0.00), fixed + expandable 3.23 (0.00). **A flat footprint comes from fixing the
SHAPES, not from the allocator.** Measure the per-clip constant under the precision you train
at: the same benchmark in fp32 reads 2.2x high.

**Why validation got slow.** `MetricsCalculator5.calculate_per`'s edit distance is an
O(decoded × target) Python double loop over torch scalars, and with no blank to collapse runs
the decoded length starts near the FRAME count. Measured at 1992 frames/clip: 6.7 s/clip
untrained, 0.68 s at 50 ms phone runs, 0.14 s at 250 ms runs. Capped by `per_max_segments` and
`per_every_n_epochs`; `per_enabled: false` is the current default.

**The `AccumulateGrad node's stream does not match` warning** is unrelated and harmless: it
fires on a single GPU too, and the two things that could hold a graph across steps were checked.

**Corpus choice matters for diagnosis.** MLS_7l_100h and mswc38_50h both degrade the English
TIMIT probe for reasons that are not the loss — mswc38 is isolated words
(`mean_phone_frames` 26.9 against libri's 15.9, so most frames are silence and the hat's entropy
is intrinsically lower). Multilingual quality is **not diagnosable locally**: every labelled
boundary set available is English, so "lost English while gaining other languages" and "degraded
outright" look identical from here. Diagnose the loss on libri_100h.

---

## Checkpoint inventory (2026-09-24)

`tmp/training/p3cupe/`: pa05a, pa07e, pa07f, pa08a, pa08b, pa09a, pa09b, pb01a, pb02a–pb02j,
plus `logs/` (every pb02 arm's stdout, `snap_sweep*.txt`, the queue logs).

- **pa07f e0 is the best checkpoint and the warm start for the final recipe.**
- `pb01a/` holds BOTH pb01a and pb01b: pb01b's files have 10:xx timestamps.
- pb02a–j: each has an e0/e1 top-2 `.ckpt` and a `final_e10.pt`. None beats pa07f. Keep them only
  as reference points for an A/B; they are safe to delete.
- pa09a / pa09b are the collapsed `soft_target_mix: 1.0` runs; keep only as a "before" arm.
- pa01d e3 was deleted in an earlier cleanup.
