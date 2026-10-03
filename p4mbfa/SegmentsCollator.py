'''
Segment dataset for p4mbfa: standard_g2p gold labels (.gs.json), read as one language group.

Expects `paths_list` to be a list of dicts with keys 'audio' (absolute) and 'srt' (absolute, or
'$/...' relative to `metadata_base`), as `splits.get_splits` returns them. Each `.srt.json` has a
`.gs.json` next to it, written by standard_g2p's `phonemize_srt`:

    {"lang": "slv", "tonal": false, "tone_layer": false, "needs_word_segmentation": false,
     "word_segmenter": null, "n_tones": 22, "n_gold_ph": 270, "n_gold_phg": 15, ...,
     "segments": [{"start": 0.0, "end": 9.6, "gold_ph": [...], "gold_phg": [...],
                   "tone": [...], "phonemes": [...], ...}]}

Stored labels are GOLD indices; `LGI.to_local` converts them to the language group's local
indices here, at load time (see check_data.py, which this reproduces).

## What raises, and what is only counted

RAISES (a contract is broken, so nothing downstream would mean anything):
  - the yaml's `n_tokens` / `n_tones` / `n_phg` disagree with standard_g2p's tables, or a file
    header disagrees with them;
  - a segment converts to a different language group or token count than the yaml's;
  - `<blank>` (0) appears in a label sequence, or a label is out of range;
  - `gold_phg` is not index-aligned with `gold_ph`, or a SIL token's group is not SIL_GROUP;
  - a segment has no `gold_ph_count`, or its tone layer does not expand onto `local_ph` with it
    (see `_tone_sequence`);
  - an audio file's sample rate is not `sample_rate`, or it is much shorter than its segment;
  - fewer than `min_kept_frac` of the files or segments survive the filters.

COUNTED AND PRINTED (the filters check_data.py defines, plus two it implies):
  - missing audio / missing .gs.json files;
  - tonal files (`skip_tonal`), unsegmented files (`skip_unsegmented`);
  - segments outside [duration_min, duration_max], with more than `max_fold_frac` of their
    tokens folded to `<unk>`, with fewer than `ph_seq_min` or more than `ph_seq_max` tokens,
    or outside [tokens_per_sec_min, tokens_per_sec_max];
  - nonzero tone values in a file whose `tone_layer` is False. These are kept - the tone term
    is masked for such clips anyway - but counted, because they are G2P artifacts (measured on
    FLEURS 'latin': 8 segments, `mri` and `ron`).
'''
import gc
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

from local_paths import get_path

sys.path.append(get_path('standard_g2p_repo'))
from standard_g2p import gold_g2p as GP                      # noqa: E402
from standard_g2p import lang_group_inventory as LGI         # noqa: E402

from p4mbfa.loss import NO_TONE, SIL_CLASS, SIL_GROUP        # noqa: E402

METADATA_BASE = get_path("metadata_base")
tmp_dir = "tmp/training"

BLANK_TOKEN = 0
PAD_LABEL = -100            # ignore_index convention; `loss.prepare_targets` clamps it
CACHE_VERSION = 2           # bump when the cached layout or the filter semantics change


def resolve_metadata_path(path):
    '''paths_list stores metadata paths as absolute, or '$/...' relative to METADATA_BASE.'''
    if path[:2] == '$/':
        return os.path.join(METADATA_BASE, path[2:])
    if path[0] != '/':
        return os.path.join(METADATA_BASE, path)
    return path


def gs_path_of(srt_path):
    srt_path = resolve_metadata_path(srt_path)
    if not srt_path.endswith('.srt.json'):
        raise ValueError(f"expected an .srt.json path, got {srt_path}")
    return srt_path[:-len('.srt.json')] + '.gs.json'


def verify_inventory(lang_group, n_tokens, n_tones, n_phg):
    '''
    The yaml's label-space sizes against standard_g2p's tables. Returns the language group's
    token list. Raises on any mismatch: a head sized from the yaml and labels sized from the
    tables would otherwise disagree silently.
    '''
    known = LGI.list_all_lang_groups
    if lang_group not in known:
        raise ValueError(f"unknown lang_group {lang_group!r}; known: {known}")
    tokens = list(LGI.tokens(lang_group))
    checks = {
        f"n_tokens ({lang_group})": (int(n_tokens), int(LGI.n_tokens(lang_group))),
        "n_tokens vs len(tokens)": (int(n_tokens), len(tokens)),
        "n_tones": (int(n_tones), int(GP.N_TONES)),
        "n_phg": (int(n_phg), int(GP.phoneme_features.N_GROUPS)),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise ValueError("yaml label spaces disagree with standard_g2p (yaml, tables): "
                         + ", ".join(f"{k}: {v}" for k, v in bad.items()))
    # `loss.py` hard-codes these indices; a regenerated table that moves them must fail here.
    if tokens[BLANK_TOKEN] != '<blank>' or tokens[SIL_CLASS] != 'SIL':
        raise ValueError(f"special tokens moved: tokens[:4] = {tokens[:4]}; loss.py expects "
                         f"<blank> at {BLANK_TOKEN} and SIL at {SIL_CLASS}")
    if GP.phoneme_features.GROUP_NAMES[SIL_GROUP] != 'special':
        raise ValueError(f"phoneme group {SIL_GROUP} is "
                         f"{GP.phoneme_features.GROUP_NAMES[SIL_GROUP]!r}, loss.py expects "
                         f"'special' (the group of SIL)")
    return tokens


def filter_paths_by_lang_group(paths_list, lang_group):
    '''Keep the files whose language belongs to `lang_group` (check_data.py's first filter).'''
    if lang_group not in LGI.list_all_lang_groups:
        raise ValueError(f"unknown lang_group {lang_group!r}; known: {LGI.list_all_lang_groups}")
    kept = [p for p in paths_list if LGI.lang_group_of(p['lang']) == lang_group]
    if not kept:
        raise ValueError(f"no file in the paths list belongs to lang_group {lang_group!r}")
    langs = Counter(p['lang'] for p in kept)
    print(f"lang_group {lang_group!r}: {len(kept)} of {len(paths_list)} files, "
          f"{len(langs)} languages: {dict(sorted(langs.items()))}")
    return kept


# Every collator setting, and the yaml key it comes from. One list, so the cache key, the
# datamodule and the train script cannot drift apart.
COLLATOR_HP_KEYS = (
    'lang_group', 'n_tokens', 'n_tones', 'n_phg',
    'duration_min', 'duration_max', 'ph_seq_min', 'ph_seq_max',
    'tokens_per_sec_min', 'tokens_per_sec_max', 'max_fold_frac',
    'skip_tonal', 'skip_unsegmented', 'min_kept_frac', 'sample_rate',
)


def collator_config_from_hp(hp):
    missing = [k for k in COLLATOR_HP_KEYS if not hasattr(hp, k)]
    if missing:
        raise KeyError(f"yaml is missing collator keys: {missing}")
    return {k: getattr(hp, k) for k in COLLATOR_HP_KEYS}


class SegmentClass:
    '''
    Parallel, growable arrays of every kept segment. Label sequences are int16 (every label
    space here is < 300 and the -100 pad fits), which keeps a few hundred tokens per row cheap
    in RAM and in the cache file.
    '''

    GROWTH_FACTOR = 1.5

    def __init__(self, ph_seq_max, init_capacity=1000):
        self.capacity = int(init_capacity)
        self.ph_seq_max = int(ph_seq_max)
        self.idx = 0

        def seqs():
            return torch.full((self.capacity, self.ph_seq_max), PAD_LABEL, dtype=torch.int16)

        self.phn_seqs = seqs()
        self.grp_seqs = seqs()
        self.tone_seqs = seqs()
        self.tone_mask = torch.zeros(self.capacity, dtype=torch.bool)
        self.phn_seqs_len = torch.zeros(self.capacity, dtype=torch.long)
        self.clip_is = torch.zeros(self.capacity, dtype=torch.long)
        self.start_samples = torch.zeros(self.capacity, dtype=torch.long)
        self.end_samples = torch.zeros(self.capacity, dtype=torch.long)

    def add(self, phn_seq, grp_seq, tone_seq, tone_mask, clip_i, start_sample, end_sample):
        n = len(phn_seq)
        if n > self.ph_seq_max:
            raise ValueError(f"sequence of {n} > ph_seq_max {self.ph_seq_max}")
        if not (len(grp_seq) == len(tone_seq) == n):
            raise ValueError(f"layer lengths differ: ph {n}, phg {len(grp_seq)}, "
                             f"tone {len(tone_seq)}")
        i = self.idx
        self.phn_seqs[i, :n] = torch.tensor(phn_seq, dtype=torch.int16)
        self.grp_seqs[i, :n] = torch.tensor(grp_seq, dtype=torch.int16)
        self.tone_seqs[i, :n] = torch.tensor(tone_seq, dtype=torch.int16)
        self.tone_mask[i] = bool(tone_mask)
        self.phn_seqs_len[i] = n
        self.clip_is[i] = clip_i
        self.start_samples[i] = start_sample
        self.end_samples[i] = end_sample
        self.idx += 1
        if self.idx == self.capacity:
            self._grow(max(self.capacity + 1, int(self.capacity * self.GROWTH_FACTOR)))

    def _arrays(self):
        return ('phn_seqs', 'grp_seqs', 'tone_seqs', 'tone_mask', 'phn_seqs_len', 'clip_is',
                'start_samples', 'end_samples')

    def _grow(self, new_capacity):
        for name in self._arrays():
            old = getattr(self, name)
            fill = PAD_LABEL if name.endswith('_seqs') else 0
            new = torch.full((new_capacity,) + tuple(old.shape[1:]), fill, dtype=old.dtype)
            new[:old.shape[0]] = old
            setattr(self, name, new)
        self.capacity = new_capacity

    def __len__(self):
        return self.idx

    def get(self, idx):
        return (self.phn_seqs[idx].long(), self.grp_seqs[idx].long(), self.tone_seqs[idx].long(),
                self.tone_mask[idx], self.phn_seqs_len[idx], self.clip_is[idx].item(),
                self.start_samples[idx].item(), self.end_samples[idx].item())

    def finalize(self):
        # .clone(): a [:idx] slice is a view, and torch.save would write the whole arena.
        for name in self._arrays():
            setattr(self, name, getattr(self, name)[:self.idx].clone())
        self.capacity = self.idx


class Segments_Collator(Dataset):
    def __init__(self, paths_list, name, lang_group, n_tokens, n_tones, n_phg,
                 duration_min, duration_max, ph_seq_min, ph_seq_max,
                 tokens_per_sec_min, tokens_per_sec_max, max_fold_frac,
                 skip_tonal, skip_unsegmented, min_kept_frac, sample_rate=16000,
                 regen_cache=False):
        print(f"Segments_Collator[{name}]: {len(paths_list)} files, lang_group {lang_group!r}")
        if not paths_list:
            raise ValueError(f"Segments_Collator[{name}]: empty paths list")

        self.name = name
        self.config = {
            'lang_group': str(lang_group), 'n_tokens': int(n_tokens), 'n_tones': int(n_tones),
            'n_phg': int(n_phg), 'duration_min': float(duration_min),
            'duration_max': float(duration_max), 'ph_seq_min': int(ph_seq_min),
            'ph_seq_max': int(ph_seq_max), 'tokens_per_sec_min': float(tokens_per_sec_min),
            'tokens_per_sec_max': float(tokens_per_sec_max),
            'max_fold_frac': float(max_fold_frac), 'skip_tonal': bool(skip_tonal),
            'skip_unsegmented': bool(skip_unsegmented), 'min_kept_frac': float(min_kept_frac),
            'sample_rate': int(sample_rate), 'cache_version': CACHE_VERSION,
        }
        self.lang_group = self.config['lang_group']
        self.n_tokens = self.config['n_tokens']
        self.n_tones = self.config['n_tones']
        self.n_phg = self.config['n_phg']
        self.ph_seq_max = self.config['ph_seq_max']
        self.sample_rate = self.config['sample_rate']
        self.wav_len_max = int(round(self.config['duration_max'] * self.sample_rate))

        # Always re-verified, cache or not: the tables can be regenerated under a cache.
        self.tokens = verify_inventory(self.lang_group, self.n_tokens, self.n_tones, self.n_phg)
        self.n_gold_ph = int(GP.phoneme_inventory_gold.N_TOKENS)

        self.cache_dir = os.path.join(tmp_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir,
                                       f'p4mbfa_collator_{self._cache_key(paths_list)}.pt')
        print(f"Cache file: {self.cache_file}")
        if not regen_cache and os.path.exists(self.cache_file):
            self._load_from_cache()
            return

        self.SegmentClass = SegmentClass(self.ph_seq_max, init_capacity=max(1024, len(paths_list)))
        self.all_clips_paths = {}
        self.unique_phones = set()
        self.total_duration = 0.0
        self.new_collator(paths_list)
        self._save_to_cache()

    # ------------------------------------------------------------------ building

    def new_collator(self, paths_list: List[Dict]):
        counts = Counter()
        examples = {}                   # first offending path per skip reason
        ph_lens, durations = [], []

        n_workers = max(1, min(32, (os.cpu_count() or 2) - 1))
        with ThreadPoolExecutor(max_workers=n_workers) as executor, \
                tqdm(total=len(paths_list), desc=f"Reading {self.name}", leave=False) as pbar:
            # `map` keeps the input order, so the dataset order is deterministic; an exception
            # inside a worker is re-raised here.
            for clip_i, (results, file_counts, file_examples) in enumerate(
                    executor.map(self._process_file, paths_list, range(len(paths_list)),
                                 chunksize=64)):
                counts.update(file_counts)
                for k, v in file_examples.items():
                    examples.setdefault(k, v)
                if results:
                    self.all_clips_paths[clip_i] = paths_list[clip_i]['audio']
                for r in results:
                    self.SegmentClass.add(*r)
                    self.unique_phones.update(r[0])
                    ph_lens.append(len(r[0]))
                    durations.append((r[6] - r[5]) / self.sample_rate)
                pbar.update(1)
        gc.collect()

        self.SegmentClass.finalize()
        self.total_duration = float(sum(durations))
        self._print_statistics(counts, examples, ph_lens, durations, len(paths_list))

    def _process_file(self, path_data, clip_i):
        '''One file -> (kept segment tuples, skip counts, first example per skip reason).'''
        cfg = self.config
        counts, examples, results = Counter(), {}, []

        def skip(reason, where, n=1):
            counts[reason] += n
            examples.setdefault(reason, where)

        audio_path = path_data['audio']
        if not os.path.exists(audio_path):
            skip('file: missing audio', audio_path)
            return results, counts, examples
        gs_path = gs_path_of(path_data['srt'])
        if not os.path.exists(gs_path):
            skip('file: missing .gs.json', gs_path)
            return results, counts, examples
        with open(gs_path) as f:
            gold = json.load(f)
        lang = gold['lang']
        n_segs = len(gold['segments'])
        counts['file: read'] += 1

        if gold['tone_layer']:
            if cfg['skip_tonal']:
                skip('file: tonal (skip_tonal)', gs_path)
                counts['segment: in skipped file'] += n_segs
                return results, counts, examples
            if not (gold['tonal'] and gold['n_tones'] > 0):
                raise ValueError(f"{gs_path}: tone_layer is set but tonal={gold['tonal']} "
                                 f"n_tones={gold['n_tones']}")
        if gold['needs_word_segmentation']:
            if cfg['skip_unsegmented']:
                skip('file: needs word segmentation (skip_unsegmented)', gs_path)
                counts['segment: in skipped file'] += n_segs
                return results, counts, examples
            if gold['word_segmenter'] is None:
                raise ValueError(f"{gs_path}: needs word segmentation but word_segmenter is None")

        header = {'n_tones': self.n_tones, 'n_gold_ph': self.n_gold_ph, 'n_gold_phg': self.n_phg}
        bad = {k: (gold[k], v) for k, v in header.items() if gold[k] != v}
        if bad:
            raise ValueError(f"{gs_path}: header disagrees with the tables (file, tables): {bad}")

        for s, seg in enumerate(gold['segments']):
            counts['segment: total'] += 1
            where = f"{gs_path}#{s}"
            duration = seg['end'] - seg['start']
            if not (cfg['duration_min'] <= duration <= cfg['duration_max']):
                skip('segment: duration', where)
                continue

            if 'gold_ph_count' not in seg:
                raise KeyError(f"{where}: no gold_ph_count; this .gs.json predates it. Rewrite it "
                               f"with phonemize_srt (or recompute with LGI.gold_ph_count_of).")
            local = LGI.to_local(seg['gold_ph'], lang, seg['gold_ph_count'])
            if local is None:
                raise ValueError(f"{where}: to_local returned None for lang {lang!r}")
            if local['lang_group'] != self.lang_group:
                raise ValueError(f"{where}: lang {lang!r} is in lang_group "
                                 f"{local['lang_group']!r}, not {self.lang_group!r}")
            if local['n_local_ph'] != self.n_tokens:
                raise ValueError(f"{where}: n_local_ph {local['n_local_ph']} != n_tokens "
                                 f"{self.n_tokens}")
            ph = list(local['local_ph'])
            phg = list(seg['gold_phg'])
            n = len(ph)
            if n == 0:
                skip('segment: too few tokens', where)
                continue

            # check_data.py divides by the VOCABULARY size (208), which lets a segment keep
            # ~10 folded tokens whatever its length. The intent is the fraction of THIS
            # segment's tokens that became <unk>, so that is what is used here.
            if local['n_folded'] / n > cfg['max_fold_frac']:
                skip('segment: too many tokens folded to <unk>', where)
                continue
            if n < cfg['ph_seq_min']:
                skip('segment: too few tokens', where)
                continue
            tps = n / duration
            if not (cfg['tokens_per_sec_min'] <= tps <= cfg['tokens_per_sec_max']):
                skip('segment: tokens per second', where)
                continue
            if n > self.ph_seq_max:
                skip('segment: more tokens than ph_seq_max', where)
                continue

            self._check_labels(where, ph, phg, seg['gold_ph'])
            tone, tone_mask, nonzero_masked = self._tone_sequence(where, gold, seg, local, n)
            if nonzero_masked:
                counts['kept: nonzero tone in a tone_layer=False file (masked)'] += 1
                examples.setdefault('kept: nonzero tone in a tone_layer=False file (masked)',
                                    where)

            start = int(round(seg['start'] * self.sample_rate))
            end = int(round(seg['end'] * self.sample_rate))
            results.append((ph, phg, tone, tone_mask, clip_i, start, end))
            counts['segment: kept'] += 1
        return results, counts, examples

    def _check_labels(self, where, ph, phg, gold_ph):
        if len(phg) != len(ph) or len(gold_ph) != len(ph):
            raise ValueError(f"{where}: gold_phg ({len(phg)}) / gold_ph ({len(gold_ph)}) / "
                             f"local_ph ({len(ph)}) are not index-aligned")
        if BLANK_TOKEN in ph:
            raise ValueError(f"{where}: <blank> ({BLANK_TOKEN}) in a label sequence")
        if min(ph) < 0 or max(ph) >= self.n_tokens:
            raise ValueError(f"{where}: local_ph out of range [0, {self.n_tokens})")
        if min(phg) < 0 or max(phg) >= self.n_phg:
            raise ValueError(f"{where}: gold_phg out of range [0, {self.n_phg})")
        # `loss.prepare_targets` pads with (SIL_CLASS, SIL_GROUP); a real SIL must agree.
        if any(g != SIL_GROUP for p, g in zip(ph, phg) if p == SIL_CLASS):
            raise ValueError(f"{where}: a SIL token has a phoneme group other than {SIL_GROUP}")

    def _tone_sequence(self, where, gold, seg, local, n):
        '''
        (tone labels index-aligned with local_ph, tone_mask, nonzero-but-masked flag).

        The tone layer is per PHONEME, and a phoneme that backs off can become several gold
        tokens (`ʈ͡ʂ` -> `ʈ ʂ`; 6% of FLEURS 'latin' segments). `LGI.expand` repeats each
        phoneme's tone over the tokens it became, using `local_ph_count` from `to_local`. It
        runs for every segment, masked or not, so a tone layer that does not match its phonemes
        raises either way.

        With `tone_layer: False` the layer is all 0 by construction and masked, so an all-NO_TONE
        row is used and any nonzero value is only reported.
        '''
        try:
            tone = LGI.expand(seg['tone'], local['local_ph_count'])
        except ValueError as e:
            raise ValueError(f"{where}: tone layer does not line up with its phonemes: {e}") from e
        if len(tone) != n:
            raise ValueError(f"{where}: expanded tone has {len(tone)} entries, local_ph has {n}")
        if not gold['tone_layer']:
            return [NO_TONE] * n, False, any(t != NO_TONE for t in tone)
        if min(tone) < 0 or max(tone) >= self.n_tones:
            raise ValueError(f"{where}: tone out of range [0, {self.n_tones})")
        return tone, True, False

    # ------------------------------------------------------------------ audio

    @torch.no_grad()
    def get_wav(self, clip_i, start_sample, end_sample):
        audio_path = self.all_clips_paths[clip_i]
        num_frames = end_sample - start_sample
        wav, sr = torchaudio.load(audio_path, frame_offset=start_sample, num_frames=num_frames,
                                  normalize=True)
        if sr != self.sample_rate:
            # frame_offset / num_frames were computed at `sample_rate`, so the segment is wrong.
            raise ValueError(f"{audio_path}: sample rate {sr}, expected {self.sample_rate}")
        # A segment that runs past the end of its file by more than 100 ms means the
        # timestamps and the audio disagree.
        if wav.shape[1] < num_frames - int(0.1 * self.sample_rate):
            raise ValueError(f"{audio_path}: segment [{start_sample}, {end_sample}) but only "
                             f"{wav.shape[1]} samples were read")

        wav = self._rms_normalize(wav.mean(dim=0))
        wav = wav[:self.wav_len_max]            # at most one sample of rounding
        wav_len = wav.shape[0]
        wav = torch.nn.functional.pad(wav, (0, self.wav_len_max - wav_len), 'constant', 0)
        return wav, wav_len

    @staticmethod
    def _rms_normalize(wav):
        rms = torch.sqrt(torch.mean(wav ** 2))
        if rms > 0:
            wav = wav / rms
        return wav

    def __len__(self):
        return len(self.SegmentClass)

    def __getitem__(self, idx):
        ph, grp, tone, tone_mask, ph_len, clip_i, start, end = self.SegmentClass.get(idx)
        wav, wav_len = self.get_wav(clip_i, start, end)
        return ph, grp, tone, tone_mask, wav.unsqueeze(0), ph_len, wav_len

    def decode(self, ph_seq):
        '''Local indices -> token strings (padding dropped).'''
        return [self.tokens[int(i)] for i in ph_seq if int(i) >= 0]

    # ------------------------------------------------------------------ cache

    def _cache_key(self, paths_list):
        cfg = json.dumps(self.config, sort_keys=True)
        paths = "\n".join(p['audio'] for p in paths_list)
        return hashlib.md5(f"{cfg}\n{len(paths_list)}\n{paths}".encode()).hexdigest()

    def _load_from_cache(self):
        size_gb = os.path.getsize(self.cache_file) / 2 ** 30
        print(f"Loading cache file ({size_gb:.2f} GB): {self.cache_file}")
        cached = torch.load(self.cache_file, weights_only=False)
        if getattr(cached, 'config', None) != self.config:
            raise ValueError(f"{self.cache_file} was built with a different config; delete it "
                             f"or pass regen_cache")
        for key in ('SegmentClass', 'all_clips_paths', 'unique_phones', 'total_duration',
                    'stats_lines'):
            setattr(self, key, getattr(cached, key))
        for line in self.stats_lines:
            print(line)

    def _save_to_cache(self):
        # Write-then-rename: every DDP rank builds the same file, and a reader must never see
        # a half-written one.
        tmp = f"{self.cache_file}.{os.getpid()}.tmp"
        torch.save(self, tmp)
        os.replace(tmp, self.cache_file)
        print(f"Saved collator cache ({os.path.getsize(self.cache_file) / 2 ** 30:.2f} GB)")

    def _print_statistics(self, counts, examples, ph_lens, durations, n_files):
        lines = [f"--- {self.name} collator: lang_group {self.lang_group!r}, "
                 f"{self.n_tokens} tokens ---"]
        for key in sorted(counts):
            lines.append(f"  {key:62s} {counts[key]:>8d}"
                         + (f"   e.g. {examples[key]}" if key in examples else ""))
        kept = counts['segment: kept']
        if kept == 0:
            for line in lines:
                print(line)
            raise ValueError(f"{self.name}: no segment survived the filters")

        def summary(label, xs):
            t = torch.tensor(xs, dtype=torch.float)
            return (f"  {label} min/median/mean/max: {t.min():.2f} / {t.median():.2f} / "
                    f"{t.mean():.2f} / {t.max():.2f}")

        lines.append(summary("tokens per segment", ph_lens))
        lines.append(summary("segment seconds   ", durations))
        lines.append(f"  distinct local tokens used: {len(self.unique_phones)} of "
                     f"{self.n_tokens} (4 are specials; <blank> is never a target)")
        file_frac = counts['file: read'] / n_files
        seg_total = counts['segment: total'] + counts['segment: in skipped file']
        seg_frac = kept / max(seg_total, 1)
        lines.append(f"  kept {kept} segments = {100 * seg_frac:.1f}% of segments in readable "
                     f"files; {100 * file_frac:.1f}% of files readable; "
                     f"{self.total_duration / 3600:.2f} hours")
        self.stats_lines = lines
        for line in lines:
            print(line)
        with open(self.cache_file + ".txt", "w") as f:
            f.write("\n".join(lines) + "\n")

        floor = self.config['min_kept_frac']
        if file_frac < floor or seg_frac < floor:
            raise ValueError(f"{self.name}: kept {100 * file_frac:.1f}% of files and "
                             f"{100 * seg_frac:.1f}% of segments, below min_kept_frac "
                             f"{floor}. See the counts above.")
