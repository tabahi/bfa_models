import os
import sys
import json




REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


sys.path.append(REPO_ROOT)
from local_paths import get_path


METADATA_DIR = get_path('metadata_base')  # e.g.  /mnt/intelpa-3/rehman/common_dir

sys.path.append(get_path('speech_data_manager_paths_manager'))
sys.path.append("/mnt/intelpa-2/rehman/codes/CharsiuG2P")


import splits

def _resolve(path, metadata_dir):
    """paths_list stores paths as absolute, or '$/...' relative to METADATA_DIR."""
    if not path:
        return path
    if path[:2] == '$/':
        return os.path.join(metadata_dir, path[2:])
    if path[0] != '/' and metadata_dir:
        return os.path.join(metadata_dir, path)
    return path


def check_g2p_import():
    # importing from above dir
    from standard_g2p.gold_g2p import (goldG2P, resolve_tag, N_TONES,
                                        has_segmenter, require_segmenter,
                                        UnsupportedLanguageError)
    from standard_g2p import lang_group_inventory as LGI

    #G = goldG2P(device='cuda:0')        # weights are fetched once into tmp/
    #d = G.phonemize_sentence('uma parceria', lang='pt-BR')        # stages 1 and 2
    #t = LGI.to_local(d['gold_ph'], 'pt-BR')   


    



def run():
    #check_g2p_import()
    
    paths_list, total_hours, val_data = splits.get_splits('fleurs', dev_mode=False)
    val_paths_list = val_data[0] if val_data else None
    
    from standard_g2p import lang_group_inventory as LGI
    selected_lang_group = 'latin'
    known_lang_groups = LGI.list_all_lang_groups
    assert selected_lang_group in known_lang_groups, f"Unknown lang group {selected_lang_group}, known: {known_lang_groups}"
    member_langs = LGI.member_langs[selected_lang_group]   # ['af', 'ang', 'arg', 'az', ...]   59 codes
    print(f"Lang group {selected_lang_group} has {len(member_langs)} member langs: {member_langs}")

    paths_list = [p for p in paths_list if LGI.lang_group_of(p['lang']) == selected_lang_group]
    val_paths_list = [p for p in val_paths_list if LGI.lang_group_of(p['lang']) == selected_lang_group] if val_paths_list else None
    print(f"Filtered paths_list to {len(paths_list)} items, val_paths_list to {len(val_paths_list) if val_paths_list else 0} items")    

    lang_group_phones = LGI.tokens(selected_lang_group)
    print(lang_group_phones)
    index_blank = lang_group_phones.index('<blank>')
    index_sil = lang_group_phones.index('SIL')
    index_noise = lang_group_phones.index('noise')
    index_unk  = lang_group_phones.index('<unk>')
    n_tokens = LGI.n_tokens(selected_lang_group)

    # now get the numbers for gold inventory (global phonemes) for all languages
    from standard_g2p import gold_g2p as GP
    n_tones = GP.N_TONES
    n_gold_ph = GP.phoneme_inventory_gold.N_TOKENS
    n_gold_phg = GP.phoneme_features.N_GROUPS

    
    # filter conditions
    min_duration = 0.5
    max_duration = 30.0
    min_tokens_per_sec = 2.0 # tps
    max_tokens_per_sec = 20.0 # tps
    skip_tonal = True
    skip_unsegmented = True

    collection = []

    segment_skipped_count = 0
    for i in range(len(paths_list[-10:])):
        audio_path = paths_list[i]['audio'] # audio path is always absolute, but srt path may be relative to METADATA_DIR
        if not  os.path.exists(audio_path):
            print(f"Audio path {audio_path} does not exist, skipping")
            continue
        srt_path = _resolve(paths_list[i]['srt'], METADATA_DIR)
        gs_path = srt_path.replace('.srt.json', '.gs.json').replace('/fleurs/', '/fleurs_gold/')
        if os.path.exists(gs_path):
            with open(gs_path) as f:
                gold_data = json.load(f)
        else:
            continue
        lang = gold_data['lang']

        if skip_tonal and gold_data['tone_layer']:
            continue
        elif not skip_tonal and gold_data['tone_layer']:
            assert gold_data['tonal'] and gold_data['n_tones'] > 0, f"Tone layer present but tonal={gold_data['tonal']} n_tones={gold_data['n_tones']}"

        if skip_unsegmented and gold_data['needs_word_segmentation']:
            continue
        elif not skip_unsegmented and gold_data['needs_word_segmentation']:
            assert gold_data['word_segmenter'] is not None, f"Word segmentation needed but word_segmenter is None"

        # check n counts match in gold_data
        assert gold_data['n_tones'] == n_tones, f"Mismatch in n_tones for lang {lang}: gold_data={gold_data['n_tones']} vs GP={n_tones}"
        assert gold_data['n_gold_ph'] == n_gold_ph, f"Mismatch in n_gold_ph for lang {lang}: gold_data={gold_data['n_gold_ph']} vs GP={n_gold_ph}"
        assert gold_data['n_gold_phg'] == n_gold_phg, f"Mismatch in n_gold_phg for lang {lang}: gold_data={gold_data['n_gold_phg']} vs GP={n_gold_phg}"
        
        for s in range(len(gold_data['segments'])):
            segment = gold_data['segments'][s]
            duration = segment['end'] - segment['start']
            if duration < min_duration or duration > max_duration:
                print(f"Skipping segment {s} because duration is {duration}")
                segment_skipped_count += 1
                continue

            
            lang_group_data = LGI.to_local(segment['gold_ph'], lang)
            #print(f"Lang {lang}, tokens: {tokens}")
            assert lang_group_data is not None, f"Failed to convert gold_ph to local tokens for lang {lang}"
            assert lang_group_data['lang_group'] == selected_lang_group, f"Lang group mismatch: expected {selected_lang_group}, got {lang_group_data['lang_group']}"
            assert lang_group_data['n_local_ph'] == n_tokens, f"Mismatch in n_local_ph for lang {lang}: expected {n_tokens}, got {lang_group_data['n_local_ph']}"
            local_tokens = lang_group_data['local_ph']

            # skip if too  many phonemes are back-offed to <unk> after lang-group conversion. 
            n_fold_ratio = lang_group_data['n_folded'] / n_tokens
            if n_fold_ratio > 0.05:
                print(f"Skipping segment {s} because too many phonemes are back-offed to <unk> after lang-group conversion: {n_fold_ratio:.2%}")
                segment_skipped_count += 1
                continue
            
            # skip segments conditions:
            if len(local_tokens) < 5:
                print(f"Skipping segment {s} because tokens is empty")
                segment_skipped_count += 1
                continue

            tokens_per_sec = len(local_tokens) / duration
            if tokens_per_sec < min_tokens_per_sec or tokens_per_sec > max_tokens_per_sec:
                print(f"Skipping segment {s} because tokens per second is {tokens_per_sec}", "len(gold_ph)=", len(segment['gold_ph']), len(local_tokens), "duration=", duration)
                segment_skipped_count += 1
                continue


            seg_obj = {
                'audio_path': audio_path,
                'start': segment['start'],
                'end': segment['end'],
                'gold_ph': segment['gold_ph'],
                'gold_phg': segment['gold_phg'],
                'local_ph' : local_tokens,
                'tone': segment['tone'],
                'stress': segment['stress'],
                'length': segment['length'],
            }
            collection.append(seg_obj)

    print("Gathered {} segments after filtering, skipped {} segments".format(len(collection), segment_skipped_count))

if __name__ == "__main__":
    run()


''' print(gold_data)

{'audio_path': '/mnt/intelpa-3/rehman/speech_data/fleurs_dev_16khz/sl_si/4926296088328531102.wav', 'lang': 'slv', 'g2p_lang': 'slv', 
'tonal': False, 'tone_layer': False, 'needs_word_segmentation': False, 'word_segmenter': None, 
'n_tones': 22, 'n_stresses': 3, 'n_lengths': 3, 'n_gold_ph': 270, 'n_gold_phg': 15, 
'segments': [{'start': 0.0, 'end': 9.6, 
'text': 'tibetanski budizem sicer temelji na naukih bude vendar se ozira tudi na mahajansko pot ljubezni in številne tehnike iz indijske joge', 
'words': ['tibetanski', 'budizem', 'sicer', 'temelji', 'na', 'naukih', 'bude', 'vendar', 'se', 'ozira', 'tudi', 'na', 'mahajansko', 
'pot', 'ljubezni', 'in', 'številne', 'tehnike', 'iz', 'indijske', 'joge'], 
'phonemes': ['SIL', 't', 'i', 'b', 'ɛ', 't', 'á', 'n', 's', 'k', 'i', 'b', 'u', 'd', 'ì', 'z', 'ə', 
'm', 's', 'ì', 't͡s', 'ɛ', 'r', 't', 'ɛ', 'm', 'é', 'l', 'j', 'i', 'n', 'á', 'n', 'a', 'ú', 'k', 'i',
 'x', 'b', 'ù', 'd', 'ɛ', 'ʋ', 'ɛ', 'n', 'd', 'à', 'r', 's', 'ɛ', 'ɔ', 'z', 'ì', 'r', 'a', 't', 'ù', 'd', 
 'i', 'n', 'á', 'm', 'a', 'x', 'a', 'j', 'á', 'n', 's', 'k', 'ɔ', 'p', 'ó', 't', 'l', 'j', 'u', 'b', 'é',
   'z', 'n', 'i', 'í', 'n', 'ʃ', 't', 'ɛ', 'ʋ', 'í', 'l', 'n', 'ɛ', 't', 'ɛ', 'x', 'n', 'ì', 'k', 'ɛ', 'í',
     's', 'ì', 'n', 'd', 'i', 'j', 's', 'k', 'ɛ', 'j', 'ó', 'ɡ', 'ɛ', 'SIL'], 
     'tone': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
         'stress': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
             'length': [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
               2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0,
                 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 
                 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                   'word_num': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                     3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 
                     9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13,
                       13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 17,
                         17, 17, 17, 17, 17, 17, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20], 
'gold_ph': [1, 9, 6, 22, 24, 9, 156, 8, 5, 11, 6, 22, 16, 15, 214, 29, 21, 14, 5, 214,
 56, 24, 13, 9, 24, 14, 157, 10, 20, 6, 8, 156, 8, 4, 229, 11, 6, 39, 22, 233, 15, 24, 
 102, 24, 8, 15, 183, 13, 5, 24, 31, 29, 214, 13, 4, 9, 233, 15, 6, 8, 156, 14, 4, 39, 4, 
 20, 156, 8, 5, 11, 31, 23, 158, 9, 10, 20, 16, 22, 157, 29, 8, 6, 163, 8, 34, 9, 24, 102, 
 163, 10, 8, 24,9, 24, 39, 8, 214, 11, 24, 163, 5, 214, 8, 15, 6, 20, 5, 11, 24, 20, 158, 28, 24, 1],
   'gold_phg': [0, 4, 1, 5, 2, 4, 3, 9, 7, 4, 1, 5, 1, 5, 1, 8, 2, 9, 7, 1, 6, 2,11, 4, 2, 9, 2, 10, 12,
     1, 9, 3, 9, 3, 1, 4, 1, 7, 5, 1, 5, 2, 12, 2, 9, 5, 3, 11, 7, 2, 2, 8, 1, 11, 3, 4, 1, 5, 1, 9, 3, 9, 
     3, 7, 3, 12, 3, 9, 7, 4, 2, 4, 2, 4, 10, 12, 1, 5, 2, 8, 9, 1, 1, 9, 7, 4, 2, 12, 1, 10, 9, 2, 4, 2, 7,
       9, 1, 4, 2, 1, 7, 1, 9, 5, 1, 12, 7, 4, 2, 12, 2, 5, 2, 0], 'gold_unmapped': []}]}


'''