from os.path import exists, join
import json
from bournemouth_aligner import Phonemizer_ph66 as Phonemizer # currently using ph66 from v2.0, it may change in the future depending on models and languages
from local_paths import get_path

METADATA_BASE = get_path("metadata_base") # meta_data_base is the base path for all metadata files, including srt and ts files. It is set in local_paths.py - It's purpose is allow moveable relative paths for meatadata files.


def phonemize_texts(paths_list, redo=False, langs =  ['en']):
    '''
    paths_list is a list of dicts, each with keys 'srt' and 'ts'
    example: [{'srt': 'path/to/srt1.json', 'ts': 'path/to/ts1.json'}, {'srt': 'path/to/srt2.json', 'ts': 'path/to/ts2.json'}]
    Where srt is the input srt file path and ts is the output ts file path. srt files are text transcriptions of audio in the whisper format.
    Example srt file:
    {
        "segments": [
            {
                "start": 0.0,  
                "end": 0.3,
                "text": "hello world",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.3, "probability": 0.9}
                ]
            }
        ],
        "text": "hello world",  
        "audio_path": "path_to_audios/en/test/2904_2350_000056.flac"
    }
    The output ts file will be in the following format:
    {
        "segments": [
            {
                "start": 0.0,
                "end": 0.3,
                "text": "hello world",
                "ph": [1, 22, 53, 1, 23, 4, 28, 14, 29, 17, 48],
                "phg": [1, 2, 4, 6, 6, 1, 1, 1, 7, 7, 9],
                "word_num": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                "words": ["hello", "world"],
                "audio_path": "path_to_audios/en/test/2904_2350_000056.flac"
            }
        ]
    }
    - If redo is False, existing TS files are not overwritten.
    '''
    print("Phonemizing...")

    P = Phonemizer(language=None, remove_noise_phonemes=True) # if remove_noise_phonemes, total phonemes will be len(phoneme_mapped_index) - 1
    SIL_token = P.phoneme_index['sil']  # This is the index of the silence phoneme in the phoneme index
    print("Using common phoneme index of vocab size:", len(P.phoneme_index))


    def phonemize_segments(srt_part, ts_path):
        '''
        srt format:
        {"segments": [{"start": 0.0, "end": 0.3, "words": [{"word": "hello", "start": 0.0, "end": 0.3, "probability": 0.9}]}, ],  text: "hello world" , "audio_path": "path_to_audios/en/test/2904_2350_000056.flac"}
        
        ts format:
        {"segments": [{"start": 0.0, "end": 0.3, "text": "hello world", "ph": [1, 22, 53, 1, 23, 4, 28, 14, 29, 17, 48], 
                    "phg": [1, 2, 4, 6, 6, 1, 1, 1, 7, 7, 9], "word_num": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
                    "words": ["hello", "world"], "audio_path": "path_to_audios/en/test/2904_2350_000056.flac"}]}

        ph: phoneme class index
        phg: phoneme group index
        word_num: word number - tracks which word each phoneme belongs to
        words: list of words that correspond to the indices in word_num
        '''
        ts_out = {"segments": [], }
        with open(srt_part, "r") as file:
            srt_data = file.read()
            srt_data = json.loads(srt_data)
            
            # Initialize the structure for ts output
            ts_out = {"segments": [], "audio_path": srt_data.get("audio_path", "")}

            # Process each segment
            for segment_idx, segment in enumerate(srt_data["segments"]):
                # Initialize tracking of which words actually have phonemes

                segment_out = {
                                    "start": segment["start"],
                                    "end": segment["end"],
                                    "text": segment["text"],
                                    "ph": [],
                                    "phg": [],
                                    "words": [],
                                    "word_num": [],

                                }
                phonemizer_output = P.phonemize_sentence(segment["text"])
                segment_out["ph"] = phonemizer_output["ph66"] 
                segment_out["phg"] = phonemizer_output["pg16"]
                segment_out["words"] = phonemizer_output["words"]
                segment_out["word_num"] = phonemizer_output["word_num"]

                if segment_out["ph"]:  # Only process if there are phonemes
                
                    # Check for trim silences if available
                    if 'trim' in segment:
                        TRIM_START_THRESHOLD = 0.25
                        TRIM_END_THRESHOLD = 0.3

                        if segment["trim"][0] > TRIM_START_THRESHOLD:
                            segment_out["ph"] = [SIL_token] + segment_out["ph"]
                            segment_out["phg"] = [SIL_token] + segment_out["phg"]
                            segment_out["word_num"] = [0] + segment_out["word_num"]
                        if segment["trim"][1] > TRIM_END_THRESHOLD:
                            segment_out["ph"] = segment_out["ph"] + [SIL_token]
                            segment_out["phg"] = segment_out["phg"] + [SIL_token]
                            segment_out["word_num"] = segment_out["word_num"] + [segment_out["word_num"][-1]]

                ts_out["segments"].append(segment_out)

        # Save the ts output to the specified path
        with open(ts_path, "w") as f:
            json.dump(ts_out, f, ensure_ascii=False)
    for lang in langs:
        P.set_backend(lang)

        for i in range(len(paths_list)):
            
            if (paths_list[i]['lang']==lang):
                srt_path = paths_list[i]['srt']
                ts_path = paths_list[i]['ts']
                if (srt_path[0:2] == '$/'): srt_path = join(METADATA_BASE, srt_path[2:])
                if (ts_path[0:2] == '$/'): ts_path = join(METADATA_BASE, ts_path[2:])
                if (not exists(ts_path)) or redo:
                    if (exists(srt_path)):
                        phonemize_segments(srt_path, ts_path)
                    else:
                        print("Error: srt file not found", srt_path)
        print("last ts file:", ts_path)
    print("Done")
