# FLEURS data per language group

Measured with the collator's filters (`p4mbfa/SegmentsCollator.py`) at the yaml defaults:
`duration_min 0.5`, `duration_max 20` s, `ph_seq_min 5`, `ph_seq_max 320` tokens,
`tokens_per_sec 2-20`, `max_fold_frac 0.05` (share of a segment's tokens folded to `<unk>`).
"Kept" is what survives them. train = `fleurs_train`, val = `fleurs_test`, no limits.
Gold inventory fingerprint 9438371ed6dd.

| group | tokens | train files | train h raw -> kept | val files | val h kept |
|---|---|---|---|---|---|
| latin | 208 | 91,497 | 307.2 -> **273.5** | 10,447 | 30.7 |
| cyrillic | 127 | 21,518 | 72.9 -> **65.1** | 2,450 | 7.1 |
| abjad | 90 | 12,483 | 43.1 -> **38.2** | 1,464 | 4.3 |
| other_alphabetic | 104 | 12,796 | 43.0 -> **39.8** | 1,555 | 4.7 |
| brahmic | 96 | 5,567 | 18.8 -> **16.8** | 889 | 2.6 |
| cjk | 48 | 5,170 | 17.0 -> **14.9** | 616 | 1.8 |
| thai_khmer (km excluded) | 38 | 2,602 | 8.5 -> **7.8** | 291 | 0.8 |
| burmese (not trainable) | 4 | 2,533 | 10.0 -> **0.0** | 240 | 0.0 |
| vietnamese | 45 | 2,994 | 9.1 -> **8.7** | 282 | 0.8 |
| japanese | 28 | 2,292 | 7.4 -> **7.0** | 278 | 0.9 |
| **total** (km excluded) | | 159,452 | 536.9 -> **471.7** | 18,512 | 53.7 |

- **burmese** cannot be trained: standard_g2p's `burmese` group holds only the four special tokens,
  so 98.5% of tokens fold to `<unk>` and every segment is filtered.
- **thai_khmer** excludes Khmer (`exclude_langs: [km]`): it is on standard_g2p's excluded list and
  folds 19.8% of its tokens to `<unk>`. Its yaml uses `min_kept_frac: 0.85`, because the Thai val
  set keeps 89.0% (30 test clips are longer than 20 s).
- **cyrillic** includes Croatian (`hr`, Latin script): standard_g2p puts hbs in cyrillic.
- The duration filter is most of every drop. `duration_max: 30` would keep 99.4% of latin
  segments (Maori `mi` loses 5.7 of its 10 h at 20 s) for 1.5x the compute per clip.

## Per language (train split)

### latin

| lang | files | h raw -> kept | `<unk>` | dropped segments |
|---|---|---|---|---|
| af | 1,032 | 3.6 -> 3.2 | 0.00% | duration 59, tokens/sec 5 |
| az | 2,665 | 9.3 -> 8.5 | 0.00% | duration 126 |
| bs | 3,091 | 10.0 -> 9.5 | 0.00% | duration 72, tokens/sec 2 |
| ca | 2,300 | 7.4 -> 7.0 | 0.00% | duration 74 |
| cs | 2,811 | 8.4 -> 8.2 | 0.00% | duration 38 |
| cy | 2,753 | 10.0 -> 6.8 | 0.00% | duration 443, tokens/sec 402 |
| da | 2,465 | 7.5 -> 7.3 | 0.00% | duration 37 |
| de | 2,987 | 9.0 -> 8.8 | 0.00% | duration 32 |
| en-US | 2,602 | 7.5 -> 7.1 | 0.00% | duration 52 |
| es-419 | 2,796 | 8.8 -> 8.5 | 0.00% | duration 41 |
| et | 2,501 | 7.3 -> 7.0 | 0.01% | duration 42, tokens/sec 1 |
| fi | 2,704 | 8.8 -> 8.2 | 0.00% | duration 89 |
| fil | 1,884 | 7.7 -> 6.0 | 0.00% | duration 262 |
| fr | 3,096 | 10.0 -> 9.7 | 0.00% | duration 53 |
| ga | 2,349 | 10.0 -> 7.3 | 0.00% | duration 410 |
| gl | 2,175 | 6.7 -> 6.5 | 0.00% | duration 34 |
| hu | 3,095 | 9.3 -> 9.0 | 0.00% | duration 53 |
| id | 2,579 | 9.1 -> 8.1 | 0.00% | duration 150 |
| is | 926 | 2.8 -> 2.7 | 0.04% | duration 23 |
| it | 3,030 | 9.0 -> 8.8 | 0.00% | duration 30 |
| lb | 2,502 | 8.5 -> 7.3 | 0.00% | duration 169 |
| lt | 2,937 | 9.8 -> 8.9 | 0.10% | duration 130 |
| mi | 1,811 | 10.0 -> 4.3 | 0.00% | duration 777, tokens/sec 1 |
| ms | 2,667 | 9.5 -> 8.3 | 0.00% | duration 194 |
| mt | 2,895 | 9.9 -> 9.1 | 0.00% | duration 122 |
| nb | 2,900 | 10.0 -> 8.8 | 0.00% | duration 182 |
| nl | 2,918 | 7.7 -> 7.6 | 0.00% | duration 14 |
| pl | 2,841 | 9.2 -> 8.8 | 0.04% | duration 60 |
| pt-BR | 2,738 | 10.0 -> 8.7 | 0.00% | duration 195 |
| ro | 2,858 | 10.0 -> 9.5 | 0.00% | duration 88 |
| sk | 1,957 | 5.9 -> 5.7 | 0.00% | duration 28 |
| sl | 2,512 | 7.8 -> 7.3 | 0.01% | duration 68, tokens/sec 2 |
| sv | 2,385 | 8.4 -> 7.3 | 0.00% | duration 169 |
| sw | 2,291 | 10.0 -> 6.8 | 0.01% | duration 461 |
| tr | 2,526 | 8.3 -> 7.7 | 0.00% | duration 98 |
| uz | 2,918 | 10.0 -> 9.1 | 0.00% | duration 144 |

### cyrillic

| lang | files | h raw -> kept | `<unk>` | dropped segments |
|---|---|---|---|---|
| be | 2,433 | 9.5 -> 7.7 | 0.00% | duration 282 |
| bg | 2,972 | 9.5 -> 8.9 | 0.01% | duration 95 |
| hr | 2,941 | 10.0 -> 9.3 | 0.00% | duration 111, tokens/sec 1 |
| kk | 2,704 | 10.0 -> 8.5 | 0.00% | duration 238 |
| mk | 2,337 | 6.8 -> 6.5 | 0.02% | duration 54 |
| ru | 2,562 | 8.1 -> 7.6 | 0.01% | duration 78 |
| sr | 2,759 | 10.0 -> 8.3 | 0.02% | duration 249, tokens/sec 1 |
| uk | 2,810 | 9.0 -> 8.5 | 0.07% | duration 91 |

### abjad

| lang | files | h raw -> kept | `<unk>` | dropped segments |
|---|---|---|---|---|
| ar | 2,103 | 6.0 -> 5.8 | 0.00% | duration 36, tokens/sec 2 |
| ckb | 2,914 | 10.0 -> 9.0 | 0.01% | duration 145, tokens/sec 3 |
| fa | 2,573 | 10.0 -> 8.2 | 0.09% | duration 270, tokens/sec 5 |
| sd | 2,784 | 10.0 -> 8.8 | 0.04% | duration 181, tokens/sec 4 |
| ur | 2,109 | 7.0 -> 6.3 | 0.00% | duration 113, tokens/sec 1 |

### other_alphabetic

| lang | files | h raw -> kept | `<unk>` | dropped segments |
|---|---|---|---|---|
| am | 2,853 | 10.0 -> 9.0 | 0.03% | duration 154, tokens/sec 5 |
| el | 3,202 | 10.0 -> 9.6 | 0.09% | duration 58, folded <unk> 3, tokens/sec 1 |
| hy | 2,943 | 10.0 -> 9.3 | 0.02% | duration 116, tokens/sec 2 |
| ka | 1,491 | 5.1 -> 4.5 | 0.04% | duration 91, tokens/sec 1 |
| ko | 2,307 | 7.9 -> 7.4 | 0.06% | duration 88 |

### brahmic

| lang | files | h raw -> kept | `<unk>` | dropped segments |
|---|---|---|---|---|
| hi | 2,120 | 6.7 -> 6.2 | 0.04% | duration 73 |
| or | 1,081 | 3.4 -> 3.2 | 0.12% | duration 31 |
| ta | 2,366 | 8.7 -> 7.4 | 0.06% | duration 188 |

### cjk

| lang | files | h raw -> kept | `<unk>` | dropped segments |
|---|---|---|---|---|
| cmn | 3,237 | 9.7 -> 9.0 | 0.40% | duration 94, folded <unk> 39, tokens/sec 1 |
| yue | 1,933 | 7.3 -> 5.9 | 0.05% | duration 201, folded <unk> 1 |

### thai_khmer

| lang | files | h raw -> kept | `<unk>` | dropped segments |
|---|---|---|---|---|
| km **(excluded)** | 1,675 | 7.1 -> 0.0 | 19.83% | duration 248, folded <unk> 1427 |
| th | 2,602 | 8.5 -> 7.8 | 0.47% | duration 98, folded <unk> 24 |

### burmese

| lang | files | h raw -> kept | `<unk>` | dropped segments |
|---|---|---|---|---|
| my | 2,533 | 10.0 -> 0.0 | 98.50% | duration 281, folded <unk> 2252 |

### vietnamese

| lang | files | h raw -> kept | `<unk>` | dropped segments |
|---|---|---|---|---|
| vi | 2,994 | 9.1 -> 8.7 | 0.10% | duration 60 |

### japanese

| lang | files | h raw -> kept | `<unk>` | dropped segments |
|---|---|---|---|---|
| ja | 2,292 | 7.4 -> 7.0 | 0.09% | duration 59, folded <unk> 5 |
