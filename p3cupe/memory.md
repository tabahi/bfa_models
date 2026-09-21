# p3cupe — decision record

Written 2026-09-18, at the end of the session that built this package. `README.md` explains
**how the thing works**; this file is the **why, what was rejected, and what is still unproven**
— the parts that do not survive in code. Read both before changing the model or the loss.

---

## Status when this was written

Run `pa01a` (`libri_100h`, `epochs: 100`, from scratch) is training on the A6000 box and the
user reports **"this model isn't learning much"**. Not yet diagnosed. **No trained p3cupe model
exists yet**, so every design claim below is reasoned from p2cupe's measurements, not validated
on this lineage.

**Check first: the learning rate, not the model.** `epochs` is a learning-rate knob — OneCycle
spreads warmup over `pct_start=0.3` of the whole run, so:

| | lr peaks | after epoch 1 / 3 / 5 / 10 |
|---|---|---|
| `epochs: 20` | epoch 6 | 3.1e-5, 1.6e-4, **2.8e-4**, 2.4e-4 |
| `epochs: 100` | **epoch 30** | 1.3e-5, 1.9e-5, **3.1e-5**, 8.4e-5 |

At `epochs: 100` the first ~10 epochs run at a tenth of `lr` and look dead. p2cupe lost 10
GPU-hours to the same trap. Measured from the scheduler; **not** confirmed against pa01a's logs.

Then, in this order — it separates "the network is not learning" from "the alignment is not
moving", which need opposite fixes:

1. `train_peak_kl` falling? If flat, the model already matches its own targets → the **alignment**
   is stuck, not the network.
2. `train_mean_phone_frames` — stuck near `clip_len / n_phones` means it never left the flat
   start. Try `align_prior_epochs: 4`.
3. `train_min_dur_frac` high — phones collapsing onto `min_phone_frames`, the degenerate hard-EM
   solution.
4. `val_timit_dip_f1_25ms` against its measured random-init floor of ~30 — at the floor, the hat
   shape never formed.

---

## pa01a post-mortem (diagnosed 2026-09-18, same day) → pa01b

**It was the alignment, not the network.** From pa01a's wandb history and its checkpoints:

- `train_min_dur_frac` 0.61 (epoch 0) → 0.80 → **0.88 from epoch 2 on**, flat thereafter.
  `val_timit_sub_f1_25ms` 13.0 → **8.3 at epoch 2** (the epoch `align_prior` hit 0) and never
  moved; that is *below* the 22.7 random-init floor. PER 0.94.
- The epoch-1 model put **99.8% of TIMIT frames on 2 classes** (ids 53, 30; each ~6% of target
  phones). Viterbi then gave 90% of phones exactly 2 frames and one phone 320 frames of a
  14-phone clip. The CE trained on that and locked it in: class-prior collapse, the textbook
  hard-EM failure on raw posteriors.
- The diagonal prior did not hold it: at σ=0.1 a one-phone drift costs ~0.01 nats/frame. It
  holds on a random model, but pa01a had learnt the class bias within ~100 steps, all at lr
  ~1.5e-5 (`epochs: 100`, see the lr trap above).
- Measured on 48 TIMIT clips, aligned f1@25 (a **plain uniform split scores 52.4**):

  | checkpoint, prior weight | raw posteriors | `class_prior_norm` |
  |---|---|---|
  | step 94, w=0 | 26.0 | 43.4 |
  | epoch 1 (collapsed), w=0 | 12.6 | 21.8 |
  | step 94, w=1, σ=0.1 | 33.2 | 46.3 |
  | step 94, w=1, σ=0.03 | 46.0 | 51.9 |

**pa01b changes**, each a knob whose off-value is pa01a:

1. `class_prior_norm: true`: Viterbi aligns on `log p(c|x) − log p̄(c)` (batch-mean
   posterior, 10% uniform smoothing), i.e. hybrid-HMM scaled likelihoods. Alignment only; the CE
   is unchanged. Also applies to `align_batch` and therefore to the TIMIT probe.
2. `align_prior_sigma: 0.03`, `align_prior_epochs: 3`, `epochs: 20` (so the flat start runs at
   a real lr: 1.6e-4 by the end of epoch 3).
3. `frame_blocks: 4` (`FrameBlock`: pre-LN per-frame MLP, zero-init residual). 5.1M → 13.5M
   params, **receptive field unchanged at 38.9 ms**. `attn_radius` (banded attention with
   relative-position bias, counted in the receptive field) exists but is off.
   Activations 1.79 → 3.53 MB/window, so `batch_size` 200→100 with `accumulate_grad_batches: 2`.
   `check_batch_feasible` now measures the model instead of using the 1.74 MB constant.

Not independently validated: (1) and (2) are measured on pa01a's posteriors; (3) is reasoned
from p2cupe's depth, not measured. If pa01b still collapses, look at `train_min_dur_frac`
first, then the capacity. Also found: `freq_attention` pools over the whole 120 ms window, so
every frame sees ~1e-4 of the window through the gate. The "38.9 ms" RF is the conv RF, not a
hard wall.

---

## pa01b result → pa01c: the segmental aligner (2026-09-18)

**pa01b**: holdout sub f1@25 peaked at **61.4 at epoch 8, then fell to 57.4 by epoch 19**
(f1@10 32 → 28): hard-EM self-training drifts once it has converged. Warm-start from the best
epoch, never `final`. Diagnosis at epoch 8: no systematic offset (median +1 ms, std 21), 22% of
phones at the 10 ms minimum (TIMIT labels: 0.3%), frame argmax matches its own alignment 45%.

**The aligner, not the model, was the cheap win.** BFA's segmental decoder ideas ported into
`loss.viterbi_align` (training targets and probe share it), each a knob, off = the old DP
(verified identical on 360 random clips; new modes optimal against brute force on 145):
soft duration (min + skippable soft frames), skippable SIL, and a flux onset bonus.
Tuned on the probe's even-index half, pa01b epoch 8, **held-out half, no retraining**:

| | f1@10 | f1@25 | f1@50 | abs_err |
|---|---|---|---|---|
| min 2, nothing else (pa01b as trained) | 33.2 | 62.0 | 79.1 | 21.5 ms |
| min 6 + 4 soft, SIL skippable, bonus w30 | 54.0 | 72.2 | 83.9 | 14.3 ms |

- **The flux bonus only works from the `feature_extractor` tap** (8.9 ms RF). Flux from the
  context convs, blocks or final embedding adds ~0 — they know what the posteriors know. Same
  lesson as p2cupe's probe: the boundary lives in the layers that cannot mix time.
- Once the bonus is on, duration settings move f1@25 by <0.5 and SIL skipping ~0 on TIMIT.
- A constant per-phone cost is a no-op (every path enters the same phones); only
  duration-dependent costs can act. `enforce_all_targets=False` (word skips) scored 67.6 vs
  70.7 forced on TIMIT's matching text — for mismatched transcripts, not a boundary gain.
- Also fixed: the hat target's partner and `refine_onsets` now use the nearest VISITED phone
  (a skipped head pad used to receive ~40% of frame 0's target).
- PER/GER off (`per_enabled: false`); `frame_acc` is the identity guard. The probe now reports
  the ODD half as `val_timit_*` (ckpt monitor) and the tuning half as `val_timit_tune_*`;
  numbers are over 200 clips, not comparable 1:1 with earlier 400-clip rows.

**Open, for pa01c**: training on flux-guided targets is a bet — the tap is trained by the same
loss, so watch `val_timit_tapflux_f1_25ms` (56.7 at pa01b e8). The bonus weight was still
rising at 45 on the tune half; the decoder at w30 leans on flux for placement and on the
posteriors for order, so the model's own boundary sense (`dip_`, ~40) is still weak. Identity
(45% frame acc, 39 ms RF) is the next lever: the `attn_radius` A/B.

**pa01c result**: holdout sub f1@25 peaked at epoch 1 (73.2) and slid to 72.2 by epoch 8 — the
decoder alone gave 72.2 on pa01b e8, so retraining on segmental targets bought ~1 point. The
learned tap flux eroded as predicted (56.6 → 54.1) while `frame_acc` rose (38.9 → 41.9%): the
front end specialises for identity and loses change sensitivity.

**→ pa01d: onset bonus from log-mel flux** (`boundary_flux_source: mel`, L2 change of 80-bin
log-mel on the 5 ms grid; no parameters, cannot drift). pa01c e1 holdout: tap 54.7/73.0/84.5 →
mel **57.4/74.8/84.9** (f1@10/25/50). Cosine on log-mel was worse (46/69); L2 keeps loudness.

**⚠ The floor moved.** Random init, holdout, pa01d decoder: f1@25 **69.9** with the mel bonus,
52.0 without. pa01c e1: 74.8 with, 71.6 without. The headline is mostly the aligner; the
probe's `nobonus_f1_25ms` is the model's own learning signal. Judge pa01d on that.

pa01d: `libri_500h` (538k segments, 408 h after filtering), warm start pa01c e1, lr 5e-5,
4 epochs, batch 108 (dev run: 44.9 GB reserved).

**pa01d result** (4 epochs): headline flat 75.8–76.1 (holdout sampling error ~±0.8), `nobonus`
73.5 → 74.4 (e2) → 74.0, `frame_acc` 38 → 43%, val_loss 2.64 → 2.43. Identity keeps improving;
placement does not follow. Use e3.

## Where the gap to MFA is (2026-09-19, `error_breakdown.py`, pa01d e3, 400 probe clips)

Same clips, same scorer: MFA 79.1 f1@25, pa01d 75.8. Per boundary type (recall@25):

- **Stops: a convention difference, not an error.** pa01d starts a stop at its BURST, MFA (and
  TIMIT's phone convention) at its CLOSURE. TIMIT labels both (`tcl`, `t`), so each aligner gets
  one of the two: closure→burst 83 vs 18. It costs the vowel after the stop (stop→vowel 47 vs 81)
  because the min duration pushes it late.
- **Real gap: gradual transitions** — vowel onsets 67 vs 86 (n=4444), lateral 57 vs 77,
  rhotic→vowel 54 vs 81, glide→vowel 51 vs 78. The flux bonus cannot see these: it HELPS abrupt
  onsets (closure→burst +40, affricates +31, stops +17) and HURTS every entry into a vowel (−6.4
  overall, stop→vowel −14). With the bonus OFF, vowel onsets are 73.6 vs MFA 86.3 — that is
  the posteriors' limit, i.e. the acoustic model.
- **Rejected:** a smaller bonus on vowel entries (`vowel_scale`, tune half): 0.5 → +0.3 (noise),
  0.25 → −1.4, 0 → −3.5. Effects are not additive per class: remove the pull on the vowel and the
  same flux peak drags the neighbouring consonant boundary onto it. Code reverted.
- Not pursued (flag, don't chase): emitting closure AND burst for stops would add ~10 recall
  points on TIMIT, but that is TIMIT's labelling convention, not MFA-style alignment.

## A/B: context for identity (pa02a, 2026-09-19) — NULL RESULT, attention not adopted

`cupe_hp_ab_ctrl.yaml` vs `cupe_hp_ab_attn.yaml`, both warm from pa01d e3, 2 epochs libri_500h,
lr 5e-5, effective batch 432. The attention arm GROWS pa01d via `warm_start_arch_overrides:
[attn_radius]` (trainer.on_load_checkpoint): radius 1 in all 4 blocks, RF 38.9 → 78.9 ms,
13.5M → 17.8M params, zero-initialised outputs — verified logit-identical to pa01d at step 0.
Attention arm runs at batch 72 × 3 (37.2 GB, 78% — the CPU-calibrated check over-predicts
attention at 88%). Decide on `nobonus_f1_25ms`, then the headline, then `error_breakdown.py`'s
vowel/lateral/rhotic/glide rows. Logs: `tmp/training/p3cupe/pa02a_{ctrl,attn}.log`.

Result (epoch 1 of each; holdout probe, then error_breakdown on all 400 clips):

| | ctrl | attn |
|---|---|---|
| val_frame_acc | 44.6% | 45.7% |
| val_loss / peak_ce | 2.361 / 1.917 | 2.315 / 1.877 |
| holdout sub f1@25 (e0 / e1) | 75.92 / 75.92 | 75.63 / 75.50 |
| holdout nobonus f1@25 (e1) | 73.83 | 74.16 |
| 400-clip f1@25, bonus 30 / 0 | 75.6 / 73.3 | 75.6 / 73.7 |
| vowel / lateral / rhotic / glide @25 (bonus 0) | 73.7 / 60.2 / 71.4 / 67.8 | 74.1 / 59.3 / 71.3 / 66.4 |

- Attention buys ~1 point of frame accuracy and lower CE, but the boundaries don't move. Every
  class row is within ±1.5 (noise), and the gradual-transition gap to MFA is unchanged.
  Epochs are 35% slower (95 vs 71 min). Not adopted.
- Neither arm beat pa01d e3 (76.0). Continued hard-EM training on the same objective has
  plateaued since pa01d e1-e3. Better phone identity (frame acc) no longer translates into
  boundary placement. The limit is the self-training target: Viterbi on the model's own
  posteriors reproduces its own boundary bias. It is not capacity or context.

## Why the boundaries miss (miss_breakdown.py, pa01d e3, 400 clips, 2026-09-19)

`miss_breakdown.py` matches phone sequences (broad class + time) instead of boundaries, so
each TIMIT phone onset lands in one bucket. Closures are merged into their release, so a stop
onset is scored at the closure. MFA is scored the same way as a reference (evaluation only).

| hit@25 per phone onset | p3 bonus 30 | p3 bonus 0 | MFA |
|---|---|---|---|
| ALL | 67.8 | 74.8 | 83.7 |
| vowel | 68.9 | 78.2 | 87.0 |
| reduced vowel after stop | 46.4 | 76.9 | 87.7 |
| rhotic / lateral / glide | 49 / 48 / 55 | 58 / 53 / 53 | 73 / 73 / 84 |

- NOT the cause:
  - Sequence mismatch: wrong_class + no_match is 3.5% for p3 vs 4.2% for MFA. The ~10% of
    stops with no match is TIMIT q and extra closures, and hits both aligners equally.
  - Path drift: far_run is 1.5% with bonus 0, and misses are spread evenly across clips.
  - The duration floor, mostly: 6.3% of TIMIT phones are shorter than 30 ms, the same limit as
    MFA's 3 x 10 ms HMM. It binds in 28% of p3's non-hit misses vs 16% of MFA's. Secondary.
- The cause is LOCAL placement:
  - near misses (25-50 ms): 14.2% vs MFA 9.6%.
  - isolated far misses: 6.0% vs 2.4%.
  - Worst at gradual transitions: glides 53 vs 84; far_single on glides 17% vs 2%.
- **The onset bonus hurts real phone onsets.** Per phone it costs 7 points overall, 9 on
  vowels and 30 on reduced vowels after stops. Far misses skew early (929 early vs 561 late):
  it pulls onsets onto bursts. Its gain in the `levenshtein_align` headline comes from TIMIT's
  extra closure->burst boundaries, which are not phone onsets in the transcript. The headline
  rewards it for the wrong reason.
- The bonus is also in the TRAINING targets (PeakLoss bonus_weight 30), so the model has been
  self-training toward burst-snapped boundaries.

## Decoder-only sweep (decoder_sweep.py, pa01d e3, 2026-09-19)

Grid on the tune half: bonus {0,1,3,10,30} x min_phone_frames {3..10} x soft {0..14}.

- The bonus is monotonically worse per phone. Each step down is better:
  30 -> 10 -> 3 -> 1 -> 0 gives 68.0 -> 70.1 -> 71.6 -> 73.4 -> 74.6.
  The lev headline moves the opposite way (75.8 -> 73.3), because it credits closure->burst.
- The duration floor is not a lever. Shorter than 6 frames always hurts (min 3 is -3 points).
  Longer plateaus: 6+6 and 6+8 give 74.8-74.9 vs 74.6 at the trained 6+4, which is noise.
  Lowering the floor to 20-25 ms, the hypothesis from the miss breakdown, is rejected.
- Holdout confirmation, per-phone phone@25 (lev f1@25):

  | setting | phone@25 | lev f1@25 |
  |---|---|---|
  | bonus 30, 6+4 (current) | 68.3 | 75.8 |
  | bonus 0, 6+4 | 75.4 | 74.1 |
  | bonus 0, 6+6 | 75.4 | 74.3 |

  Per class, bonus 0 vs 30: vowel 78.8 vs 68.9, rhotic 61 vs 50, stop 77 vs 67, nasal 84 vs 79,
  lateral 51 vs 48. Glide is the only class that prefers the bonus (51 vs 53); fricative is flat.
- Conclusion: decode with bonus 0. The remaining gap to MFA's 83.7 is the model's local
  placement at gradual transitions (glides, laterals, rhotics). The next test is training
  targets built without the bonus.

## pa03a: training targets without the bonus (2026-09-19) — WORSE, declined every epoch

| holdout | pa01d e3 (start) | pa03a e0 | e1 | e5/final |
|---|---|---|---|---|
| nobonus lev f1@25 | 74.0 | 72.1 | 71.6 | 69.8 |
| per-phone phone@25 (bonus 0) | 75.4 | 74.5 | 74.2 | 72.6 |
| sub f1@10 | ~58 | 42.7 | 41.4 | 39.3 |

- Meanwhile val_frame_acc rose 44.6% -> 53.0%, val_loss fell 2.36 -> 1.99, and
  val_min_dur_frac went 0.118 -> 0.202: many more phones squeezed to the 6-frame floor.
  The model is agreeing better with its OWN alignments while real boundaries get worse.
  This is hard-EM drift; frame_acc is self-agreement, not accuracy.
- Final per class, bonus 0: rhotic up (61 -> 64); vowel, nasal and fricative down 3-5; glide and
  lateral flat. Near misses grew (14.2% -> 16.6%).
- Every warm-started continuation has declined: pa01b after e2, pa01d's later epochs, both
  pa02a arms, pa03a. The bonus in the targets SLOWED the drift (it is a fixed external
  anchor); removing it sped it up. The best model is still pa01d e3 DECODED with bonus 0.
- Also tested, decode-only on pa01d e3: the bonus on obstruent entries only (pg16 6-11, with
  and without SIL). Vowels recover, but stop onsets are dragged to the burst (stop 41-59 at
  w 10-30). Never beats bonus 0 (74.6 tune). Code reverted.
- `last.ckpt` is the last MONITORED checkpoint, not the final epoch; the final weights are
  `m_<exp>_final_e<N>.pt` (same format, loadable by the analysis scripts).

## pa03b: pa03a + stronger duration prior, 1 epoch (2026-09-19) — mixed

soft_phone_frames 8, soft_frame_cost 2.0 in both training alignment and decode, bonus 0,
libri_100h, warm from pa01d e3.

- Holdout, bonus 0:
  - phone@25: 75.1 (start 75.4, pa03a e0 74.5); lev 72.0 (start 74.0).
  - Per class: lateral 51 -> 57, rhotic 61 -> 65, fricative 81 -> 78; vowel, glide, nasal and
    stop roughly flat.
  - sub f1@10: 45.5 -> 39.8; precision is falling.
- min_dur_frac 0.166 (start 0.118): the squeeze is reduced but not stopped.
- Reading: removing the bonus from the targets helps the gradual classes but loses sharpness,
  and the duration prior only slows the squeeze.
- Proposed next: pa03c = bonus 30 in the training targets (anchor) + soft 8 / 2.0, judged
  decoded with bonus 0. If it doesn't beat 75.4 per phone, stop warm-continuing pa01d.
- Gotcha: a dev run writes `m_<exp>_final_e1.pt` under the same experiment name, so don't
  wait on that file to detect the end of a real run. Wait for the process to exit instead.

## pa03c: bonus 30 in the targets + soft 8 / 2.0, 1 epoch (2026-09-19) — below start

- min_dur_frac 0.107 (start 0.118): the squeeze is STOPPED.
- frame_acc 43.7%: no self-agreement inflation.
- The boundaries are still slightly worse:
  - phone@25, bonus 0: holdout 74.3, tune 74.1 (start 75.4 / 74.6).
  - lev nobonus 73.1 (start 74.0).
  - Holdout per class: rhotic 62 and lateral 55 keep part of pa03b's gain; nasal 81 (from
    84), glide 49, fricative 81 (flat).
- Conclusion: the squeeze was a symptom, not the cause. Every variant of the target
  recipe (bonus on/off, duration prior, attention, more data, same recipe continued) ends at
  or below pa01d e3. pa01d e3 is this hard-EM loop's fixed point, and a different target
  recipe just moves it to a nearby, no-better point.
- STOP warm-continuing pa01d with target tweaks. Best model: pa01d e3 decoded with bonus 0
  (75.4 per phone holdout, MFA 83.7). The next lever is HOW the targets are built.

## Soft targets / soft-EM (2026-09-19) — implemented, pa04a running

- `loss.alignment_occupancy`: forward-backward on the SAME segmental decoder. The shared
  setup is `_segmental_topology`, also used by viterbi_align. Occupancy = d logZ / d emission
  (autograd; no hand-written backward pass).
- Verified on pa01d TIMIT posteriors:
  - refactored Viterbi bit-identical on 16/16 clips (with bonus, without, and with prior);
  - occupancy rows sum to 1;
  - scale 50 -> argmax == Viterbi on 99.75-100% of frames;
  - mix 0 loss, gradients and stats identical to the old code.
- Sharpness: at scale 1 the occupancy is nearly one-hot (mean peak 0.95 bonus 0, 0.99 bonus
  30). Posterior-mean decoding (decoder_sweep.py --occupancy-scale) holds the Viterbi
  per-phone score down to scale 0.3 (74.7 vs 74.6 tune), then degrades (0.1: 72.9; 0.05: 69.7).
  Scale 1 posterior-mean decoding gives 74.9 (noise-level over Viterbi).
- Cost: 1.6 s per 108 x 2000-frame batch at chunk 128 (Viterbi 1.1 s), +1.4 GB. The dev run
  peaked at 37.1 GB allocated / 45.4 reserved; nvidia-smi shows 47.5 GB in training.
- hp: `soft_target_mix` (0 = hat only), `soft_target_scale` (default 0.3).
- pa04a (`cupe_hp_pa04a.yaml`) = the pa02a_ctrl recipe (bonus 30, soft 4 / 1.0) +
  soft_target_mix 1.0, scale 0.3; libri_100h, 1 epoch, warm from pa01d e3. Judge per phone
  with bonus 0 against 75.4 holdout / 74.6 tune.

### pa04a result — soft-EM does not help; the misses are CONFIDENT

- Standard metrics were stable for once: holdout headline 75.9 (start 76.0), nobonus 73.8
  (74.0), sub f1@10 58.9 (58.6). min_dur_frac 0.122 (no squeeze).
- Per phone, bonus 0: holdout 74.1 (start 75.4), tune 74.3 (start 74.6). Vowel 76.3 vs 78.8 and
  glide 46.6 vs 51 on holdout. Not better.
- Caveat: at scale 0.3 the target entropy was 0.09, LOWER than the hat's 0.17, so pa04a mostly
  tested "no ramp", not "keep the uncertainty".
- The decisive diagnostic (scratch `onset_confidence.py`, pa01d e3, 400 clips, bonus 0):
  - Each onset's posterior std comes from the occupancy CDF.
  - At scale 1, 66% of far misses (>50 ms) have std < 10 ms (vowels 71%, rhotics 75%,
    laterals 74%). The median std is 0 ms even for near misses.
  - At scale 0.3 the std grows with the error (7.6 / 10.9 / 16.9 ms for hit / near / far) but
    stays ~13 ms for vowel far misses, far short of the error.
  - The model is confidently wrong, so no re-weighting of its own beliefs can move those
    boundaries. Soft-EM, like every earlier target tweak, is capped here.
- Remaining levers are in the acoustic evidence itself (user to choose):
  - speaker normalisation (MFA's speaker-adapted second pass);
  - context-dependent / transition modelling (MFA's triphone states), since we have one class
    per phone at gradual transitions.
- The code stays: soft_target_mix defaults to 0 (off); verified and documented in loss.py.

## Cleanup (2026-09-19, after commit 8961c4d)

- Removed the experiment configs cupe_hp_ab_ctrl / ab_attn / pa03a / pa03b / pa03c / pa04a.yaml
  (recoverable from 8961c4d).
- Deleted the checkpoints of pa02a_*, pa03a-c and pa04a (their results are all recorded here).
- Kept pa01a-d; pa01d e3 is the warm-start source and the current best.
- Tools kept: error_breakdown.py, miss_breakdown.py, decoder_sweep.py.

## Sub-phone states (pa05a, running 2026-09-19) — user-requested, may be rolled back

The user asked for MFA's "tri-phone strategy": 2 states for short phonemes, 3 for long, and a
loss that punishes out-of-order states. It was tried before with model3a and did not work, but
the model is now very different.

NB this is MFA's multi-STATE phone topology, not context-dependent triphones.

- `states.py` (StateInventory):
  - 185 state classes. SIL and noise get 1 state; stops (pg16 6, 7), glides (15), h and the
    tap get 2; everything else 3. SIL is state 0.
  - `expand` turns phone sequences into state sequences, so the unchanged segmental decoder
    aligns states.
  - `collapse` turns a state alignment back into phones: a phone's onset is its first state's.
  - `phone_log_probs` = log-sum-exp over a phone's states.
  - `order_penalty` = mass on state k of phone p at t times mass on its states j < k at
    t + lag (lags 1-2).
  - Self-test: in order 0, reversed 0.4.
- Model: `phone_states` config key; classes_dim = 185.
- Trainer:
  - `forward` expands targets and adds order_loss_weight * order penalty (logs `order_mass`).
  - `align_batch` aligns states and collapses them to phones (refinement reads phone
    posteriors). The probe's dip readout uses phone posteriors.
  - The duration knobs come from state_min_frames / state_soft_frames (2 / 1).
- Warm start: `phone_states` is the one CONVERTIBLE key in warm_start_arch_overrides. The
  classifier's phone rows are copied to their states with the bias lowered by log(n_states).
  Verified: the state model's phone posteriors equal pa01d's (max diff 7.6e-6), and the dip
  readout is identical.
- Step-0 probe with 100 clips: f1@25 74.8 vs pa01d 75.6. The states start identical, so the
  split between them is arbitrary; the dev run gives nobonus 73.7.
- pa05a (`cupe_hp_pa05a.yaml`): the pa01d recipe + phone_states, libri_100h, 2 epochs (the new
  head needs steps to separate its states), monitor nobonus. Judge per phone with bonus 0
  against 75.4 holdout / 74.6 tune.
- Rollback: set `phone_states: false` (the default). Phone mode is unchanged (smoke test passes).

### pa05a result — FIRST gain over pa01d per phone, and still improving with training

- Standard probe at the trained 2+1: nobonus 73.3 (pa01d 74.0), headline 74.0. The dip
  readout fell to 38.6 (43.3).
- order_mass 0.083 -> 0.051 (val); min_dur_frac 0.34 (per STATE, not comparable).
- Per phone, bonus 0 (decoder_sweep --min/--soft are PER STATE here):

  | | tune | holdout |
  |---|---|---|
  | pa01d best decode | 74.9 | 75.4 |
  | pa05a at the trained 2+1 | 75.4 | 75.7 |
  | pa05a at 3+1 (best on tune) | 76.5 | 76.2 |

  - 3+2 is equal; 3+3 and up slowly lower; min 4 collapses to 72.
  - A longer floor does NOT explain it: pa01d at 8-10 phone frames gets 74.5 / 73.3.
- Epoch 1 -> 2 on tune at 3+2: 76.1 -> 76.4. Improving, unlike every hard-EM continuation.
- Holdout per class at 3+1 vs pa01d:
  - better: vowel 81.5 vs 78.8, lateral 56.3 vs 51.3, stop 78.4 vs 76.8, rhotic 62.0 vs 60.9;
  - worse: nasal 79.2 vs 84.0, fricative 79.6 vs 81.1;
  - flat: glide ~50 (MFA 84).
- Proposed next, pa05b: continue from pa05a e2 with state_min_frames 3 / soft 1 so training
  matches the best decode. Watch nasals (a 2-state nasal is a one-line change).

### pa05b: pa05a e2 + TRAINING at 3+1 per state, 2 epochs — WORSE, drift is back

- Per phone at 3+1: tune 76.5 (start) -> 76.0 (e1) -> 75.5 (e2); holdout 76.2 -> 75.5.
- nobonus 73.4 -> 73.3; min_dur_frac 0.34 -> 0.40 (state-level squeeze); order_mass 0.051 -> 0.047.
- Nasals keep falling: pa01d 84.0 -> pa05a 79.2 -> pa05b 76.3 (tune). Vowels and laterals keep
  most of their gain.
- BEST so far: pa05a e2 (`pa05a/m_pa05a_epoch=1_step=500_...73.3032.ckpt`) decoded at
  state_min_frames 3 / soft 1: 76.2 holdout, 76.5 tune per phone (MFA 83.7).
- Proposed: pa05c = pa05a repeated with NASALS as 2-state phones (12% of TIMIT nasals are
  < 30 ms). If it doesn't beat pa05a, stop and make pa05a e2 + 3/1 decoding the recipe.

### ROLLED BACK (2026-09-19, user)

The user judged every post-commit change worthless and asked to return to single-state phones.
- model.py, trainer.py, timit_probe.py and decoder_sweep.py were restored from 8961c4d.
- states.py, cupe_hp_pa05a/b.yaml and the pa05a/b checkpoints were deleted.
- The pa05 notes above are the only record left.

## pa06: wider window + per-frame context, English vs multilingual (running 2026-09-19)

User's question: is more context bad for MULTILINGUAL performance? MLS_7l_100h has no English.
The user chose: warm start from pa01d e3, no 120 ms baseline runs, and widen the per-frame
context as well as the window.

- Why both: the per-frame receptive field was 38.9 ms whatever the window. Windows are
  processed independently and stitched; only `freq_attention` (a per-window channel gate)
  pools the whole window.
- trainer: `GEOMETRY_KEYS` (window_size_ms, window_stride_ms) are now allowed in
  `warm_start_arch_overrides`. No weight depends on the window, so pa01d loads unchanged; the
  model config's input_wav_length follows the yaml.
- pa06a (libri_100h) and pa06b (MLS_7l_100h) are identical apart from data_src:
  - window 360 / stride 180, attn_radius 4 in all 4 blocks: RF 198.9 ms, 17.75M params;
  - 2 epochs, lr 5e-5, bonus 30 (pa01d recipe), monitor nobonus;
  - batch 72 x accumulate 3 x 2 GPUs = 432. Batch 54 measured 33.6 GB reserved, although the
    startup check said 95%: it over-predicts attention again. 72 measured 44.7 GB (93%).
- Step 0 (the dev run, before any training): re-windowing alone costs ~2 points. Headline
  74.8 vs pa01d 76.0; nobonus 71.9 vs 74.0. The per-window gate now pools 360 ms, and the
  stitch changed.
- Judge on the TIMIT probe (English) for both, per phone with bonus 0 (decoder_sweep). pa06b
  is then an English test of an MLS-trained model, i.e. cross-lingual.

### pa06 result and cleanup (user, 2026-09-19)

- pa06a (libri, 360 ms + attention) was not much better than 120 ms. pa06b (MLS) was stopped
  after 1 epoch at 156 min/epoch; holdout sub f1@10 55.3, f1@25 71.6.
- Decision: stay at 120 ms windows. The pa06 configs, checkpoints and trainer window override
  are removed; trainer.py is back to 8961c4d.
- MLS_7l_100h is ~6.5x libri_100h per epoch: 1614 vs 250 optimiser steps at an effective
  batch of 432.

## pa07a: the pa04a recipe on MLS, 10 epochs (master cupe_hp.yaml, 2026-09-19)

- The user picked pa04a as best from the wandb table (probe headline). Its checkpoints had been
  deleted in the cleanup, so the recipe was recreated from `git show
  8961c4d:modules/p3cupe/cupe_hp_pa04a.yaml`.
- cupe_hp.yaml is now that recipe with data_src MLS_7l_100h, epochs 10, warm from pa01d e3:
  soft_target_mix 1.0, scale 0.3, bonus 30, 120 ms windows, batch 108 x 2, monitor nobonus.
- Dev run OK: 45.5 GB reserved; the step-0 probe equals pa01d.
- Expected time: ~2.4 h/epoch (pa04a's ~5.4 s/step x 1614 steps), so ~24 h for 10 epochs.
  The user starts it.

### f1@10 vs MFA — check before using it as a selling point (pa01d e3, same 400 clips)

| | all TIMIT boundaries @10/25/50 | without closure->burst @10/25/50 |
|---|---|---|
| p3 bonus 30, sub-frame (probe readout) | 58.9 / 75.7 / 85.2 | 52.1 / 71.8 / 85.7 |
| p3 bonus 0, sub-frame | 46.8 / 73.2 / 85.6 | 46.8 / 75.0 / 87.9 |
| MFA | 53.6 / 79.1 / 88.5 | 57.7 / 83.4 / 91.7 |

- MFA's 53.6 on these 400 clips matches its full-TIMIT 53.65, so the comparison is sound on
  the standard scorer: p3 leads by +5.3 at @10.
- The lead comes from TIMIT's extra closure->burst boundaries, which the flux bonus places
  within 10 ms. Scored on phone onsets only, MFA leads at every tolerance (57.7 vs 52.1 at
  @10).
- The claim holds on the standard benchmark (every aligner scored the same way), but it rests
  on the burst convention.

### pa03a setup (for reference)

`cupe_hp_pa03a.yaml`: warm from pa01d e3, libri_100h (user: faster to iterate), 6 epochs
(~1/5 the steps per epoch of 500h), lr 5e-5, boundary_bonus_weight 0, ckpt_monitor
val_timit_nobonus_f1_25ms. Everything else is as in pa02a_ctrl. Starting point: nobonus 74.0
holdout; per-phone 75.4 (decoder_sweep.py). Judge the result with decoder_sweep.py /
miss_breakdown.py (bonus 0), especially glide/lateral/rhotic. Checkpoint differences are small,
so read the trend rather than the pick (user). Log: `tmp/training/p3cupe/pa03a.log`.

---

## The loss — what was chosen and why

**The end goal is MFA-style boundary alignment from sequence labels only.** So the model is
trained to make its *own posteriors carry the boundary*, rather than to emit a label sequence
from which a boundary must afterwards be recovered.

- **Hat-shaped target, no blank.** Each phone's posterior peaks at 1.0 across its interior and
  falls linearly to **exactly 0.5 at each boundary**, where the neighbour rises through 0.5. The
  boundary is therefore the **crossing point of two posteriors** — continuous, recoverable by
  interpolation (`refine_onsets`), which is why 5 ms frames are not the resolution limit.
- **Why not CTC.** CTC marks a phone with a spike and fills the gaps with blank; *where inside a
  blank run the boundary lies is not represented in the output at all*. p2cupe measured placement
  inside a blank run at chance (26.8 ms observed vs 25.9 ms predicted by random placement in a
  4.85-frame run). Snapping onto spectral flux, a learned boundary head and a partition CE were
  all attempts to recover information the loss had already discarded, and all three measured out
  at ~zero or negative. Here there is nothing to recover.
- **Targets come from the model itself**: monotonic Viterbi against the known phoneme sequence,
  under `no_grad`, every step — classic hard-EM, the bootstrap HTK/MFA use. Minimum phone
  duration is enforced by expanding each phone into `min_phone_frames` sub-states.
- **Flat start = the diagonal band prior**, weight 1 → 0 over `align_prior_epochs`. At weight 1
  the alignment is "divide the clip evenly between the phones". **This is the only schedule in
  the objective** — deliberately, after p2cupe drowned in ramps. Set to 0 when warm starting.
- **Optional SIL at both ends** (MFA's optional silence). Without a blank nothing absorbs leading
  silence, and TIMIT clips open with ~130 ms of it, so the first phone would otherwise be dragged
  to frame 0 on every clip.
- **The soft target has entropy**, so the CE cannot reach 0. `train_peak_kl` (CE minus target
  entropy) is the number that goes to zero; the raw loss plateaus around 0.2–0.4 by construction.
  This also prevents the saturation that made p2cupe's posteriors useless for placement.
- **The group head is the same loss at coarser granularity**, at a fixed `group_loss_weight`. It
  is *not* a second objective: no schedule, no ramp, no dynamic priority controller.

**Knobs, in the order worth turning:** `ramp_frames` (3.0 — the identity/placement trade-off; if
`sub_f1` does not beat frame-grid `f1`, this is the knob, **not** the architecture) →
`align_prior_epochs` → `context_kernels` → `min_phone_frames` → `group_loss_weight`.

---

## The model — what was chosen and why

- **CNN only, no transformer.** p2cupe's own per-layer probe scored boundary placement at every
  tap, and the ordering was stable across checkpoints: layers that *cannot* mix across time
  scored ~20 f1 points above layers that can (`spectral_stream` 1x1 > `fusion` >
  `temporal_stream` at 150 ms RF > transformer output). A module that smears 150 ms into one
  frame cannot say where a 10 ms event starts, and no loss downstream of it can undo that.
- **5 ms frames** (conv strides 4·4·5 = 80 samples), replacing 13.125 ms. 13 ms put a ~6.5 ms
  quantisation floor under every boundary before the model did anything.
- **38.9 ms receptive field**, set by two stride-1 context convs (`context_kernels: [5, 3]`), each
  kernel adding (k−1)·5 ms. Near the 44.7 ms tap that scored best in p2cupe. Widening this walks
  back toward the 150 ms time-mixer that destroyed p2cupe's boundaries — do not, without reading
  `val_timit_dip_f1_25ms`.
- **Every conv has an odd kernel with `padding = kernel // 2`**, which is what puts frame `f` on
  input sample `f * 80` with no offset term anywhere. An even kernel shifts every timestamp by
  half a frame and nothing raises — `model.py`'s self-test asserts this with a click at a known
  sample. **Keep that test.**
- **Windowing kept** (120 ms window / 80 ms stride) because removing context is what makes the
  model multilingual — 120 ms is too short to memorise any language's phonotactics. The stride
  advances exactly 16 of 24 frames, so the stitched grid *is* the conv grid and
  `time_ms = frame * 5.0` is exact.
- **5.14M params**, down from ~30M. Output dims stay 67/17; index 66/16 ('noise') exists in the
  shared mapper and downstream tooling indexes by it, but is **never a target** here.

---

## Deleted from p2cupe, with the reason (do not reintroduce without new evidence)

| removed | why |
|---|---|
| transformer | the worst layer in its own boundary probe |
| CTC + blank + blank/SIL penalties | the boundary is not representable in its output |
| `BoundaryHead` + BCE | supervised by Viterbi onsets, i.e. distilled from a *worse* teacher than raw flux (54.4 vs 59.5 on the same checkpoint) |
| flux hinge (`bnd_lambda`) | inert at low weight, destructive at high (66.2 → 40.4 f1@25 in two epochs) |
| partition CE + lambda ramp | mechanism worked perfectly (`direct_transition_frac` 0 → 0.98) and moved the metric not at all, three rounds running |
| CTC anneal, dual-loss priority controller, curriculum imbalance | schedules tuning schedules |
| confusion matrices, confidence accumulator, contextual refiner | diagnostics for a model that no longer exists |

---

## Evaluation strategy

- **`metrics.py` is a verbatim copy of p2cupe's and must stay that way** — it is the unchanged
  yardstick. Only how much is fed to it may change (`per_max_segments`, `per_every_n_epochs`).
- **PER/GER are guards, not targets.** In p2cupe they were *anti*-correlated with placement (the
  best alignment ever recorded came from a GER 0.22 model, not the GER 0.098 one), so
  `ckpt_monitor` is a boundary f1 and `early_stop_metric` is `val_loss`. Expect PER **well above
  1** early: with no blank, every argmax flicker is an insertion.
- **The TIMIT probe is the real instrument**, rewritten for this lineage with four readouts:
  `(none)` frame-grid, **`sub_`** (interpolated crossing — the headline), **`dip_`** (free
  peak-pick on the confidence dip, *no transcript* — the falsification test for whether the V
  shape formed at all), `flux_` (p2cupe's best signal, the bar to clear).
- **Measured floors at random init** (400 clips, this architecture): decoder 22.5, sub 22.7, dip
  29.9, flux 37.1 f1@25. Everything proposes roughly as many boundaries as exist on clips with one
  every ~80 ms, so guessing scores well. **Read the gaps, not the absolutes.**

---

## Traps already hit — do not reintroduce

- **Sequences are padded with `-100`** by `SegmentsCollator`. The DP gathers emissions for every
  column, so a negative id is an out-of-bounds index → a CUDA device-side assert several kernels
  later, pointing somewhere else. Clamped once in `prepare_targets`.
- **The padding shortcut must include the window that overlaps the tail** (`(max_frames-1) //
  stride_frames + 1`, not "enough windows to reach it"). One short changes the value of the last
  ~8 frames of every clip, silently. `smoke_test.py` asserts the shortcut is numerically identical
  to computing everything.
- **p2cupe's TIMIT probe fed audio at 1/10 the training level** (`wav / rms * 0.1` against the
  collator's `wav / rms`). BatchNorm hides most of it. p3's probe matches the collator exactly.
- **Stitching accumulates in float32** even under bf16 autocast: the boundary readout interpolates
  the *difference* between adjacent frames' logits, and bf16's ~0.06 resolution at logit 10 lands
  directly on that quantity.

---

## The risk to keep in mind

The targets come from the model's own alignment, so **the loss can be driven to zero by a
segmentation that is perfectly self-consistent and systematically wrong** — every boundary 20 ms
early, say. Nothing inside the objective can detect that; it is the known failure mode of hard-EM
and the price of having no boundary labels. The three guards are all external: the diagonal prior
while the model is random, the TIMIT probe every epoch (labels training never sees), and
`train_mean_phone_frames` against the corpus's real phone rate.

**Watch the probe, not the loss.**

## pa07a result (MLS_7l_100h, 4 epochs, user ran it overnight 2026-09-19/20)

English TIMIT holdout, per epoch: sub f1@25 74.45 / 73.13 / 72.02 / 71.05; nobonus 71.44 /
67.94 / 66.41 / 65.19; sub f1@10 57.88 / 56.69 / 55.65 / 54.88. Monotonic decline, and much
steeper than any English continuation (those lost 1-2 points).

Per phone (holdout, bonus 0, min 6 / soft 4): pa01d e3 75.4 -> pa07a e1 71.4 -> pa07a e4 63.8.
Every class falls: vowel 78.8 -> 71.5 -> 63.5, rhotic 60.9 -> 49.0 -> 41.9, stop 76.8 -> 75.1
-> 68.2, fricative 81.1 -> 82.2 -> 72.1. val_frame_acc 0.367 (on MLS, against its own
alignments); val_min_dur_frac 0.119, so no squeeze; epoch_mins 160.

NOT DIAGNOSABLE with what is here: every labelled boundary set available is English (TIMIT,
Buckeye). MLS and the UCLA Phonetic Corpus have no timestamps, so "good for multilingual"
cannot be measured locally - only English degradation can, and it is large. Either the model
lost English while gaining other languages, or it degraded outright; the two look identical
from here. Deciding this needs a non-English labelled set or an external eval.

## Engineering notes moved out of the README (2026-09-20)

Kept here so the README stays about the model. All still true.

### Why two DDP ranks report different GPU memory

Not different work: plain Lightning ddp, every log sync_dist=True. The peak is set by the
longest clip in a batch when `fixed_window_batch: false`, and the caching allocator never
returns memory, so each rank strands a different amount. MEASURED, 30 steps, 1-10 s clips:

| | reserved | swing |
|---|---|---|
| ragged shapes, stock allocator | 4.03-4.46 GB | 0.43 GB |
| ragged shapes, expandable_segments | 3.10-3.14 GB | 0.04 GB |
| fixed shapes, stock allocator | 3.95 GB | 0.00 GB |
| fixed shapes, expandable_segments | 3.23 GB | 0.00 GB |

So a flat footprint comes from fixing the SHAPES, not from the allocator. Measure the per-clip
constant under the precision you train at: the same benchmark in fp32 reads 2.2x high.
`expandable_segments` is 18% leaner but maps pages on demand, so on a full card it warns and
falls back instead of raising. Not a limit though it looks like one: the windowed forward
flattens to [B*W, ...] and passes 2^31 elements around batch 187 at 10 s; a conv producing
2.21e9 elements was tested and runs fine.

### Why validation got slow, and what caps it

`MetricsCalculator5.calculate_per`'s edit distance is an O(decoded x target) Python double loop
over torch scalars. With no blank to collapse runs, the decoded length starts near the FRAME
count. MEASURED at 1992 frames/clip: 6.7 s/clip untrained, 0.68 s at 50 ms phone runs, 0.14 s
at 250 ms runs. So it shrinks as the model sharpens, but the early epochs are the expensive
ones. Capped by `per_max_segments` (per rank) and `per_every_n_epochs`; `per_enabled: false`
switches it off entirely, which is the current default. metrics.py is never touched.

### The `AccumulateGrad node's stream does not match` warning

Unrelated and harmless: it fires on a single GPU too, and the two things that could hold a
graph across steps were checked (Lightning recursive_detaches logged tensors; nothing keeps
`loss` or the alignment dict alive past the step).
