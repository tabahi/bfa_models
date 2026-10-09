# p4mbfa

A context-free CNN phoneme encoder and forced aligner for many languages: p3cupe's trunk and
segmental aligner, trained on [standard_g2p](https://github.com/tabahi/CharsiuG2P) labels with
one **fixed** cross-lingual head and two **replaceable** per-language-group heads.

```bash
python p4mbfa/smoke_test.py                                  # CPU, synthetic, ~1 min
python p4mbfa/train_mbfa.py p4mbfa/mbfa_hp_<group> [key=value ...]
python p4mbfa/export_hf.py export <checkpoint> <out_dir>     # one release: safetensors + config.json
python p4mbfa/export_hf.py release                           # every model in hf_release.yaml
python p4mbfa/export_hf.py push [--dry-run] [--only NAME]    # upload them to the Hugging Face Hub
```

What each yaml setting does, and what raising or lowering it changes, is commented in
[`mbfa_hp.yaml`](mbfa_hp.yaml) (`ramp_frames`, the aligner's duration and onset-bonus settings,
the soft onsets, the cold-start prior, regularisation). The per-group yamls share its keys.

## Using a released model

```python
from p4mbfa.inference import MbfaAligner

aligner = MbfaAligner.from_pretrained("Tabahi/mbfa-latin")   # or a release dir / training checkpoint
wav = aligner.load_audio("clip.wav")                 # mono, resampled to 16 kHz
segs = aligner.align(wav, ["SIL", "h", "ɛ", "l", "o", "SIL"])   # local tokens of the group
out = aligner.encode(wav)                            # log posteriors: out['ph'|'phg'|'tone']
```

Tokens come from text through standard_g2p (`goldG2P.phonemize_sentence`, then
`lang_group_inventory.to_local`); `config.json` lists each group's token strings. `MbfaAligner`
runs `clip.py` - the same windowing, stitching, onset-bonus flux and alignment readout training
and validation use - with the decoder settings stored in `config.json`, and `smoke_test.py`
asserts it reproduces the trainer exactly. `model.MbfaExtractor` is the per-window network alone,
for callers that do their own windowing (BFA).

## Published models

One Hugging Face repo per language group, built from [`hf_release.yaml`](hf_release.yaml) (which
lists the source checkpoint, why it was chosen and its metrics). Each repo holds
`model.safetensors` + `config.json` + a model card, and `ckpt/` with the training checkpoint for
fine-tuning (`ckpt_path:`); `from_pretrained` downloads only the first two.

Selection rule: never a checkpoint trained without noise; the most training noise wins, ranked
on validation within that level; plus one best-on-TIMIT latin model for benchmarks (TIMIT is
clean speech, so it favours low-noise training).

| repo | run, epoch | noise | data | selected on |
|---|---|---|---|---|
| `Tabahi/mbfa-latin` | ma04a, 6 | 0.05 | 273.5 h | TIMIT f1@25ms 71.22 tune / 71.70 holdout, best at noise 0.05 |
| `Tabahi/mbfa-latin-benchmark` | ma02a, 8 | 0.02 | ~170 h (62% sample) | TIMIT f1@25ms 73.33 tune / 72.93 holdout, best of all latin runs |
| `Tabahi/mbfa-cyrillic` | mr01a, 10 | 0.02 | ~40 h (62% sample) | FLEURS val_loss 2.652 |
| `Tabahi/mbfa-cjk` | mc01a, 13 | 0.02 | ~9 h (62% sample) | FLEURS val_loss 2.596 |
| `Tabahi/mbfa-japanese` | mj01a, 8 | 0.02 | 7.0 h | FLEURS val_loss 2.349 + frame accuracy |
| `Tabahi/CUPE-3-en` (p3cupe, ph66) | pa07f, 0 | 0.01 | LibriSpeech 500 h | TIMIT f1@25ms 76.10 tune / 75.58 holdout |

other_alphabetic, abjad, brahmic, thai_khmer and vietnamese have yamls but no trained model yet.

## The model

| part | output | trained on |
|---|---|---|
| trunk | 512-d embedding per 5 ms frame; 120 ms windows, 38.9 ms receptive field | p3cupe's architecture; latin (ma02a) from random weights, every other group from ma02a's final trunk |
| `phg_head` (fixed) | 15 gold phoneme groups, the same in every language group | `gold_phg` |
| `ph_head` (replaceable) | the language group's local tokens, specials included | `local_ph` |
| `tone_head` (replaceable) | 22 tone values (0 = none), the same everywhere | `tone`, masked where the file has no tone layer |

All three heads are trained off ONE alignment of `local_ph` (the segmental Viterbi in `loss.py`,
with a log-mel onset bonus), as p3cupe's peak loss: a hat target per phone, crossing 0.5 at each
boundary, so boundaries are read at sub-frame precision from the posterior crossing. Token
indices 0-3 are `<blank> SIL noise <unk>` in every group; `<blank>` is never a target.

## Language groups

One yaml per group. A new group starts from a trained checkpoint with `reset_fine_heads: true`:
the trunk and `phg_head` are kept, `ph_head` and `tone_head` are rebuilt for the yaml's
`lang_group` / `n_tokens` / `n_tones`.

| yaml | group | tokens | fleurs train h (kept) | tone head | notes |
|---|---|---|---|---|---|
| `mbfa_hp.yaml` | latin | 208 | 273.5 | masked | TIMIT probe (English) selects checkpoints |
| `mbfa_hp_cyrillic.yaml` | cyrillic | 127 | 65.1 | masked | includes `hr` (standard_g2p puts hbs here) |
| `mbfa_hp_other_alphabetic.yaml` | other_alphabetic | 104 | 39.8 | masked | |
| `mbfa_hp_abjad.yaml` | abjad | 90 | 38.2 | masked | |
| `mbfa_hp_brahmic.yaml` | brahmic | 96 | 16.8 | masked | |
| `mbfa_hp_cjk.yaml` | cjk | 48 | 14.9 | trained | segmented (cmn longest match, yue pycantonese) |
| `mbfa_hp_vietnamese.yaml` | vietnamese | 45 | 8.7 | trained | |
| `mbfa_hp_thai_khmer.yaml` | thai_khmer | 38 | 7.8 | trained | `exclude_langs: [km]`, `min_kept_frac: 0.85` |
| `mbfa_hp_japanese.yaml` | japanese | 28 | 7.0 | masked | segmented (fugashi + unidic-lite) |

Burmese has no yaml: standard_g2p's `burmese` group holds only the four specials. Per-language
numbers are in [data.md](data.md).

## Files

| file | role |
|---|---|
| `train_mbfa.py` | entry point; refuses ambiguous weight sources (`ckpt_path` vs `trunk_init_path`) |
| `mbfa_hp*.yaml` | every setting; no key has a code default, so a missing key raises |
| `SegmentsCollator.py` | `.gs.json` -> `LGI.to_local` -> segments; tone aligned with `gold_ph_count` + `LGI.expand`; filters counted, contract breaks raise |
| `datamodule.py` | Lightning datamodule; train/val overlap raises |
| `model.py` | `MbfaCNN` (trunk + 3 heads), `MbfaExtractor` |
| `loss.py` | `PeakLoss` and the segmental Viterbi; SIL = 1; maskable tone term |
| `clip.py` | clip forward, mel flux, alignment readout - shared by training and inference |
| `trainer.py` | `ComposedModel`: strict checkpoint rules, `reset_fine_heads`, `grow_frame_blocks`, `load_trunk` |
| `inference.py` | `MbfaAligner`: Lightning-free loading, encoding and forced alignment |
| `export_hf.py` | training checkpoint -> `model.safetensors` + `config.json` + model card; `release` / `push` for the Hub |
| `hf_release.yaml` | the published models: source checkpoint, why it was chosen, metrics, Hub name |
| `windowing.py` | slice + stitch (unchanged from p3cupe) |
| `timit_probe.py` / `timit_probe_prep.py` | per-epoch TIMIT boundary f1 (latin only) and its cache builder |
| `check_data.py` / `sample_gs.gs.json` | the data exploration the collator reproduces, and one `.gs.json` |

## Checkpoint rules

| yaml | effect |
|---|---|
| `trunk_init_path: <p3cupe or p4mbfa ckpt>` | trunk only; all heads fresh |
| `ckpt_path`, `reset_fine_heads: false` | full load; fine heads must match `lang_group`/`n_tokens`/`n_tones`, else raise |
| `ckpt_path`, `reset_fine_heads: true` | trunk + `phg_head` loaded; `ph_head`/`tone_head` rebuilt |
| `grow_frame_blocks: true` | a checkpoint with fewer `frame_blocks` loads; added blocks start as the identity |

Trunk shapes must otherwise match exactly; regularisation (dropouts, noise) comes from the yaml.

## Data

Paths come from `splits.get_splits(data_src, train_limit, val_limit)` (your own path manager,
see the root README). Each `.srt.json` needs a `.gs.json` beside it, written by standard_g2p's
`phonemize_srt`. Keep `train_limit` / `val_limit` null: `get_splits` samples them from all of
fleurs BEFORE the language-group filter, so a limit removes the same fraction of every group.
