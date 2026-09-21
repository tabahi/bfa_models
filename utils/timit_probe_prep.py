'''
Build the static TIMIT cache the in-training alignment probe reads.

## Why a static cache

The point of the probe is turnaround: a real TIMIT f1@25 every validation epoch instead of
waiting for the offline `preprocessing_tasks4.py` -> `alignment_tests.py` pipeline.  The
expensive and non-deterministic part of that pipeline is espeak phonemization, so it is done
once, here, and frozen.

Just as importantly, this keeps TIMIT **structurally out of the training data pipeline**.
Training stays LibriSpeech-only so the comparison with MFA is fair; TIMIT is only ever read by
the validation callback, from this file.

Routing TIMIT through `SegmentsCollator` / `.ts` files instead would not work today anyway:
  - `.ts` files do not exist for the TIMIT `all` split;
  - `_phonemeizer.phonemize_segments` does not carry `boundaries` through to the `.ts`;
  - it reads `word["word"]` while TIMIT SRTs write `{'w': ...}`.

## What is in the cache

Ground truth comes from the SRT, whose segments carry
`['start', 'end', 'text', 'words', 'phonemes', 'boundaries']`, where `boundaries` is a sorted
list of **milliseconds relative to the segment start** - exactly the frame
`alignment_tests.check_method_alignment` scores in.  TIMIT clips are a single segment starting
at 0, so segment-relative and clip-absolute coincide.

The target phoneme sequence is produced by the **same phonemizer BFA uses**
(`bournemouth_aligner_/ipamappers/ph66_phonemeizer.py`), so the probe aligns the same text
phonemization the offline pipeline would.

Usage:
    /root/anaconda3/bin/python modules/p2cupe/timit_probe_prep.py [n_clips] [out_path]
'''

import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from local_paths import get_path

PATHS_LIST = get_path("timit_paths_list")
# The `all` list is the one whose SRTs actually carry `boundaries`, but it tags every entry
# `set: 'all'`, so the test split has to be identified by clip_id against this list.
TEST_PATHS_LIST = get_path("timit_test_paths_list")
DEFAULT_OUT = "tmp/data/timit_probe_400.json"
DEFAULT_N = 400
SEED = 42


def build_cache(paths_list=PATHS_LIST, out_path=DEFAULT_OUT, n_clips=DEFAULT_N,
                test_only=True, seed=SEED, lang='en-us'):
    '''
    Args:
        paths_list: the TIMIT `all` paths cache (6300 clips, one segment each)
        out_path: where to write the probe cache
        n_clips: how many clips to keep. The offline eval runs 6300; a few hundred is enough
            for an epoch-level signal and keeps the probe to seconds of compute.
        test_only: restrict to TIMIT's own test split (1680 clips). The probe is a
            model-selection signal, so it should not be read off training-adjacent material.
        seed: fixed, so every run of every experiment scores the identical clip set.
    Returns:
        The list of cache entries that was written.
    '''
    from bournemouth_aligner.ipamappers.ph66_phonemeizer import Phonemizer

    with open(paths_list) as f:
        entries = json.load(f)

    if test_only:
        with open(TEST_PATHS_LIST) as f:
            test_ids = {e['clip_id'] for e in json.load(f)}
        entries = [e for e in entries if e.get('clip_id') in test_ids]
        print(f"Restricted to the TIMIT test split: {len(entries)} clips")
    entries.sort(key=lambda e: e['srt'])          # deterministic order before sampling
    random.Random(seed).shuffle(entries)

    P = Phonemizer(language=None, remove_noise_phonemes=True)
    P.set_backend(lang)

    cache, skipped = [], 0
    for entry in entries:
        if len(cache) >= n_clips:
            break
        srt_path, audio_path = entry['srt'], entry['audio']
        if not (os.path.exists(srt_path) and os.path.exists(audio_path)):
            skipped += 1
            continue

        with open(srt_path) as f:
            srt = json.load(f)
        segments = srt.get('segments') or []
        if len(segments) != 1:
            skipped += 1
            continue
        seg = segments[0]

        boundaries = sorted({int(round(b)) for b in seg.get('boundaries', [])})
        text = (seg.get('text') or '').strip()
        if not boundaries or not text:
            skipped += 1
            continue

        phonemized = P.phonemize_sentence(text)
        ph = list(phonemized['ph66'])
        phg = list(phonemized['pg16'])
        if not ph:
            skipped += 1
            continue

        cache.append({
            'clip_id':       entry.get('clip_id', os.path.basename(srt_path)),
            'audio':         audio_path,
            'start':         float(seg['start']),
            'end':           float(seg['end']),
            'duration':      float(seg['end']) - float(seg['start']),
            'text':          text,
            'ph':            ph,
            'phg':           phg,
            'boundaries_ms': boundaries,
        })

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(cache, f)

    n_bnd = sum(len(c['boundaries_ms']) for c in cache)
    n_ph = sum(len(c['ph']) for c in cache)
    dur = sum(c['duration'] for c in cache)
    print(f"Wrote {len(cache)} clips to {out_path} (skipped {skipped})")
    print(f"  {n_bnd} ground-truth boundaries, {n_ph} phonemized targets "
          f"(ratio {n_ph / max(n_bnd, 1):.3f}), {dur / 60:.1f} min of audio")
    print(f"  mean {n_bnd / max(len(cache), 1):.1f} boundaries and "
          f"{n_ph / max(len(cache), 1):.1f} target phones per clip")
    return cache


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    build_cache(out_path=out, n_clips=n)
