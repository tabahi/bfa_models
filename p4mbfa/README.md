# p4mbfa

p3cupe's CNN aligner, retargeted to standard_g2p labels: one **fixed** cross-lingual head and two
**replaceable** per-language-group heads.

```
/root/anaconda3/bin/python p4mbfa/train_mbfa.py p4mbfa/mbfa_hp [key=value ...]
/root/anaconda3/bin/python p4mbfa/smoke_test.py          # CPU, synthetic, ~1 min
```

## Files (same layout as p3cupe)

| file | role | vs p3cupe |
|---|---|---|
| `train_mbfa.py` | entry point | val held out by stride (fleurs' splits give train == val); weight-source checks |
| `mbfa_hp.yaml` | every setting; no key has a code default | label-space and filter keys added |
| `SegmentsCollator.py` | `.gs.json` -> `LGI.to_local` -> segments | new label format, check_data.py's filters, raises on contract breaks |
| `datamodule.py` | Lightning datamodule | train/val overlap raises |
| `model.py` | `MbfaCNN`: p3cupe trunk + 3 heads | heads replaced |
| `loss.py` | `PeakLoss`, segmental Viterbi | SIL = 1, tone term with per-clip mask, `soft_target_mix` removed |
| `trainer.py` | `ComposedModel` | strict checkpoint rules, `reset_fine_heads`, `load_trunk`; EMA/aug/repr-flux/local-delta/PER dropped (all off in pc01a) |
| `windowing.py` | slice + stitch | unchanged |
| `timit_probe.py` | per-epoch TIMIT boundary f1 | reads a local-token cache; errors propagate |
| `timit_probe_prep.py` | builds that cache from p3cupe's (same clips, re-phonemized) | new |
| `check_data.py` | the data exploration the collator reproduces | yours |

## Heads

| head | target | size | on a language-group switch |
|---|---|---|---|
| `phg_head` | `gold_phg` | `n_phg` = 15 | kept (same label space everywhere) |
| `ph_head` | `local_ph` | `n_tokens` (208 latin, includes `<blank> SIL noise <unk>`) | rebuilt |
| `tone_head` | `tone` | `n_tones` = 22 | rebuilt |

All three are trained off ONE alignment of `local_ph`. The tone term is masked for clips whose
`.gs.json` says `tone_layer: false` (all of latin), so on latin it is exactly 0 while the head
stays in the graph for DDP.

## Weights

| yaml | effect |
|---|---|
| `trunk_init_path: <p3cupe or p4mbfa ckpt>` | trunk only; all heads fresh |
| `ckpt_path: <p4mbfa ckpt>`, `reset_fine_heads: false` | full load; fine heads must match `lang_group`/`n_tokens`/`n_tones`, else raise |
| `ckpt_path: <p4mbfa ckpt>`, `reset_fine_heads: true` | trunk + `phg_head` loaded; `ph_head`/`tone_head` rebuilt for the yaml |

Trunk shape keys must equal the checkpoint's in every case (raise otherwise).

Switching to a new group, e.g.:
`experiment=mb01a lang_group=cyrillic n_tokens=127 reset_fine_heads=true ckpt_path=<latin ckpt> trunk_init_path="" timit_probe_enabled=false ckpt_monitor=val_loss ckpt_mode=min`
