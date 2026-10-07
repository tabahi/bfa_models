I

**Data per language group (fleurs).** "Kept" means what survives the collator's filters at `duration_max: 20`.

| group | tokens | train files | train h (raw → kept) | val files | val h kept | languages |
|---|---|---|---|---|---|---|
| latin | 208 | 91,497 | 307.2 → **273.5** | 10,447 | 30.7 | 36 |
| cyrillic | 127 | 21,518 | 72.9 → **65.1** | 2,450 | 7.1 | be bg hr kk mk ru sr uk |
| other_alphabetic | 104 | 12,796 | 43.0 → **39.8** | 1,555 | 4.7 | am el hy ka ko |
| abjad | 90 | 12,483 | 43.1 → **38.2** | 1,464 | 4.3 | ar ckb fa sd ur |
| brahmic | 96 | 5,567 | 18.8 → **16.8** | 889 | 2.6 | hi or ta |
| cjk | 48 | 5,170 | 17.0 → **14.9** | 616 | 1.8 | cmn yue |
| vietnamese | 45 | 2,994 | 9.1 → **8.7** | 282 | 0.8 | vi |
| thai_khmer | 38 | 4,277 (2,602 th) | 15.6 → **7.8** | 539 (291 th) | 0.8 | th (km excluded) |
| japanese | 28 | 2,292 | 7.4 → **7.0** | 278 | 0.9 | ja |
| burmese | 4 | 2,533 | 10.0 → **0** | 240 | 0 | my, can't be trained |
| **total** | | **161,127** | **544.0 → 471.7** | **18,760** | **53.7** | |

The duration filter accounts for almost all drops elsewhere. In latin it costs Māori (`mi`) the most: 10.0 h → 4.3 h, with `ga`, `sw` and `cy` close behind. `duration_max: 30` would recover most of that, at 1.5× the compute per clip.

**New yamls** (`p4mbfa/mbfa_hp_<group>.yaml`, all starting from latin ma02a with `reset_fine_heads: true`):

| group | experiment | tone head | segmentation |
|---|---|---|---|
| other_alphabetic | mo01a | masked | not needed |
| abjad | mp01a | masked | not needed |
| brahmic | mb01a | masked | not needed |
| thai_khmer | mt01a | **trains** (Thai is tonal) | pythainlp |
| vietnamese | mv01a | **trains** | not needed |

**Two group-specific settings:**
- **thai_khmer excludes Khmer.** Khmer is on standard_g2p's excluded list, and 19.8% of its tokens fold to `<unk>`, so all its segments would be filtered and the collator would raise. I added an explicit `exclude_langs` yaml key (empty in every other yaml); the filter prints how many files each excluded language removes.
- **thai_khmer uses `min_kept_frac: 0.85`.** Its 291-file val set keeps 89.0%, just under 0.9, because 30 Thai test clips are longer than 20 s. Train keeps 95.3%.

**Burmese has no yaml.** standard_g2p's `burmese` group contains only the four special tokens, so 98.5% of tokens fold to `<unk>` and nothing is left to train. That has to be fixed in standard_g2p's inventory first.



- **Epochs.** cjk, cyrillic and japanese all hit their best `val_loss` at epochs 9–13 of 30, right after the learning-rate peak at epoch 9. The best checkpoint is kept either way. On the 8 h groups (thai, vietnamese, japanese), about 20 epochs would likely reach the same result in two-thirds of the time.