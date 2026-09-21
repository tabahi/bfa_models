'''
This library keeps track of the paths to the audio files and their metadata.
 It is used to generate the paths lists for training and validation.

 The expected structure of the paths lists is as follows (for .json files listed below):
    `[
        {
            "audio": "/path/to/audio/file.wav",
            "srt": "/path/to/subtitle/file.srt.json", # whisper output (input required for the phonemizer)
            "ts": "/path/to/timestamps/file.ts.json", # output from the Phonemizer 'data_prep_1.py' script
            "vs": "/path/to/viseme/file.vs.json", # output from the batch aligmment (optional)
            "lang": "en",
            "duration": 4.0
        },  
        ... 
    ]`

Whatever speech-text dataset you have, 
first step is always to standardize the text labels in the whisper json format (srt.json).


'''
import json


paths_cache = "/tmp/speech_data_manager/paths_cache/" # set this to your own path if you want to cache the paths lists somewhere else. The default is /tmp/speech_data_manager/paths_cache/


dataloader_ready_paths_sets = {
    "libri_dev_clean": paths_cache + "paths_list_libri_dev-clean.json",
    "libri_train_100": paths_cache + "paths_list_libri_train-clean-100.json",
    "libri_train_360": paths_cache + "paths_list_libri_train-clean-360.json",
    "libri_train_500": paths_cache + "paths_list_libri_train-other-500.json",
    "libri_test_clean": paths_cache + "paths_list_libri_test-clean.json",
}

def get_assorted_paths(set_label, exclude_langs=None, include_langs=None, limit_clips=None):
    
    import random
    random.seed(42)
    
    # if set_label is a list:
    if isinstance(set_label, list):
        paths_list = []
        total_hours = 0
        for s in set_label:
            paths_list_, total_hours_ = get_assorted_paths(s, exclude_langs, include_langs, limit_clips)
            paths_list.extend(paths_list_)
            total_hours += total_hours_
        random.shuffle(paths_list)
        return paths_list, total_hours
    
    assert(set_label in dataloader_ready_paths_sets), "Unknown set_label: " + set_label + ". Available sets: " + str(dataloader_ready_paths_sets.keys())
    
    with open(dataloader_ready_paths_sets[set_label], "r") as f:
        paths_list = json.load(f)
    
    
    if (include_langs is not None):
        paths_list = [p for p in paths_list if p["lang"] in include_langs]

    if (exclude_langs is not None):
        paths_list = [p for p in paths_list if p["lang"] not in exclude_langs]
    
    random.shuffle(paths_list)
    
    if (limit_clips is not None):
        paths_list = paths_list[:limit_clips]
    
    count_lang_durations = {}
    
    for p in paths_list:
        lang = p["lang"]
        if lang not in count_lang_durations:
            count_lang_durations[lang] = 0
        if "duration" not in p:
            p["duration"] = 4
        count_lang_durations[lang] += p["duration"]
    
    for lang in count_lang_durations:
        print(f"{lang}:\t{count_lang_durations[lang]/3600:.2f} hr")

    total_hours = sum(count_lang_durations.values())/3600
    print(f"Total:\t{total_hours:.2f} hr for {len(count_lang_durations)} langs and {len(paths_list)} wavs\t Set: {set_label}")
    return paths_list, total_hours

