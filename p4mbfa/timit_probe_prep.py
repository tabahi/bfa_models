'''
Build the TIMIT probe cache in a language group's LOCAL tokens.

    /root/anaconda3/bin/python p4mbfa/timit_probe_prep.py [lang_group] [src_cache] [out_path]

The probe force-aligns the clip's phone sequence and scores the onsets against TIMIT's
hand-labelled boundaries, so it needs the sequence in the model's own dictionary. The p3cupe cache
(`tmp/data/timit_probe_400.json`, built by `utils/timit_probe_prep.py`) carries ph66 ids, which
mean nothing to a p4mbfa head. Its clips, audio paths, text and ground-truth `boundaries_ms` are
reused as they are; only the phone sequence is regenerated, once, with the same G2P that labelled
the training data (standard_g2p stage 1-2, then `LGI.to_local` stage 3), and frozen here.

TIMIT is English, so this only makes sense for the language group that contains `en` ('latin').
Any other group raises.

The output is a dict, not a bare list, so the probe can check it was built for the model it is
scoring:

    {"lang": "en", "lang_group": "latin", "n_tokens": 208, "gold_fingerprint": "...",
     "source_cache": "...", "clips": [{"clip_id", "audio", "start", "end", "duration", "text",
                                       "local_ph", "gold_phg", "n_folded", "boundaries_ms"}]}
'''

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from local_paths import get_path  # noqa: E402

sys.path.append(get_path('standard_g2p_repo'))

LANG = 'en'
DEFAULT_SRC = "tmp/data/timit_probe_400_latin.json"


def default_out(lang_group):
    return f"tmp/data/timit_probe_400_{lang_group}.json"


def build_cache(lang_group='latin', src_path=DEFAULT_SRC, out_path=None, device='cuda:0'):
    from standard_g2p import lang_group_inventory as LGI
    from standard_g2p.gold_g2p import goldG2P

    if LGI.lang_group_of(LANG) != lang_group:
        raise ValueError(f"TIMIT is {LANG!r}, which is in lang_group "
                         f"{LGI.lang_group_of(LANG)!r}, not {lang_group!r}")
    out_path = out_path or default_out(lang_group)
    with open(src_path) as f:
        src = json.load(f)

    G = goldG2P(device=device)
    n_tokens = int(LGI.n_tokens(lang_group))
    clips, fingerprint = [], None
    for entry in src:
        d = G.phonemize_sentence(entry['text'], lang=LANG)
        if d['gold_unmapped']:
            raise ValueError(f"{entry['clip_id']}: unmapped phonemes {d['gold_unmapped']}")
        t = LGI.to_local(d['gold_ph'], LANG)
        if t['lang_group'] != lang_group or t['n_local_ph'] != n_tokens:
            raise ValueError(f"{entry['clip_id']}: to_local gave {t['lang_group']} / "
                             f"{t['n_local_ph']}, expected {lang_group} / {n_tokens}")
        if len(t['local_ph']) != len(d['gold_phg']):
            raise ValueError(f"{entry['clip_id']}: local_ph and gold_phg lengths differ")
        if fingerprint is None:
            fingerprint = t['gold_fingerprint']
        elif t['gold_fingerprint'] != fingerprint:
            raise ValueError("gold_fingerprint changed within one run")
        clips.append({
            'clip_id': entry['clip_id'],
            'audio': entry['audio'],
            'start': entry['start'],
            'end': entry['end'],
            'duration': entry['duration'],
            'text': entry['text'],
            'local_ph': list(t['local_ph']),
            'gold_phg': list(d['gold_phg']),
            'n_folded': int(t['n_folded']),
            'boundaries_ms': entry['boundaries_ms'],
        })

    out = {'lang': LANG, 'lang_group': lang_group, 'n_tokens': n_tokens,
           'gold_fingerprint': fingerprint, 'source_cache': src_path, 'clips': clips}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f)

    n_bnd = sum(len(c['boundaries_ms']) for c in clips)
    n_ph = sum(len(c['local_ph']) for c in clips)
    n_fold = sum(c['n_folded'] for c in clips)
    print(f"Wrote {len(clips)} clips to {out_path}")
    print(f"  {n_bnd} ground-truth boundaries, {n_ph} {lang_group} tokens "
          f"(ratio {n_ph / max(n_bnd, 1):.3f}), {n_fold} folded to <unk>")
    ex = clips[0]
    print(f"  e.g. {ex['text']!r} -> {LGI.decode(ex['local_ph'], LANG)}")
    return out


if __name__ == "__main__":
    group = sys.argv[1] if len(sys.argv) > 1 else 'latin'
    src = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SRC
    out = sys.argv[3] if len(sys.argv) > 3 else None
    build_cache(group, src, out)
