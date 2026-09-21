'''
This model expacts path_list to be a list of dicts with keys 'audio' and 'ts' that point to the audio and ts files.
ts files are json files with the following structure:
            {"segments": [{"start": 0.0, "end": 0.3, "text": "hello world", "ph": [1, 22, 53, 1, 23, 4, 28, 14, 29, 17, 48], 
                    "phg": [1, 2, 4, 6, 6, 1, 1, 1, 7, 7, 9], "word_num": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
                    "words": ["hello", "world"], "audio_path": "path_to_audios/en/test/2904_2350_000056.flac"}]}
'''
import json
import os
import torch
import torchaudio
import gc
from typing import List, Dict
import hashlib
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset
import multiprocessing as mp
from collections import deque
from tqdm import tqdm
from local_paths import get_path

METADATA_BASE = get_path("metadata_base")
tmp_dir = "tmp/training"

machine_dir = os.path.dirname(METADATA_BASE)
    
def peak_normalize(waveform):   #2nd best
    # Normalize peak amplitude to 1
    return waveform / torch.max(torch.abs(waveform))

def rms_normalize(waveform, target_dB=-20.0): # best
    # Calculate current RMS
    rms = torch.sqrt(torch.mean(waveform ** 2))
    # Convert target dB to linear
    target_rms = 10 ** (target_dB / 20.0)
    # Calculate scaling factor
    scaling_factor = target_rms / (rms + 1e-6)  # add small epsilon to avoid division by zero
    return waveform * scaling_factor

def mean_var_normalize(waveform):   # worse
    # Normalize to zero mean and unit variance
    mean = torch.mean(waveform)
    std = torch.std(waveform)
    return (waveform - mean) / (std + 1e-6)  # add small epsilon to avoid division by zero

def per_utterance_normalize(waveform): # 2nd worse 
    # Remove DC offset
    waveform = waveform - torch.mean(waveform)
    # Scale to unit variance
    waveform = waveform / (torch.std(waveform) + 1e-6)
    return waveform




class SegmentClass:

    # Arena growth factor. 1.5 keeps the final over-allocation under ~50% while holding
    # total copy work to ~3x the row count.
    GROWTH_FACTOR = 1.5

    def __init__(self, ph_seq_max, init_capacity=1000, fill_value=None, have_boundary_labels=False):
        
        self.capacity = init_capacity
        self.ph_seq_max = ph_seq_max
        self.phn_seqs = torch.zeros((self.capacity, self.ph_seq_max), dtype=torch.long, device='cpu') #can use either torch or np
        self.grp_seqs = torch.zeros((self.capacity, self.ph_seq_max), dtype=torch.long, device='cpu')
        self._fill_value = fill_value
        if fill_value is not None:
            self.phn_seqs.fill_(fill_value)
            self.grp_seqs.fill_(fill_value)
        self.have_boundary_labels = have_boundary_labels

        if (self.have_boundary_labels):
            self.bnd_seqs = torch.zeros((self.capacity, self.ph_seq_max+10), dtype=torch.long, device='cpu')

        self.phn_seqs_len = torch.zeros((self.capacity), dtype=torch.long, device='cpu')
        self.clip_is = torch.zeros((self.capacity), dtype=torch.long, device='cpu')
        self.start_frames = torch.zeros((self.capacity), dtype=torch.long, device='cpu')
        self.end_frames = torch.zeros((self.capacity), dtype=torch.long, device='cpu')
        self.idx = 0

    def add(self, phn_seq, grp_seq, clip_i, start_frame, end_frame, boundaries=None):
        idx = self.idx
        assert (len(phn_seq) <= self.ph_seq_max), f"Phoneme sequence is too long: {len(phn_seq)} > {self.ph_seq_max}"
        assert (len(phn_seq) == len(grp_seq)), f"Phoneme sequence and group sequence should have the same length. {len(phn_seq)} != {len(grp_seq)}"
        self.phn_seqs[idx, :len(phn_seq)] = torch.LongTensor(phn_seq)
        self.grp_seqs[idx, :len(grp_seq)] = torch.LongTensor(grp_seq)
        if (self.have_boundary_labels):
            #print(phn_seq, boundaries)
            assert(boundaries is not None), "Expecting boundaries labels."
            #assert(len(phn_seq) <= len(boundaries)), f"Phoneme sequence and boundaries should have the same length. {len(phn_seq)} != {len(boundaries)}"
            self.bnd_seqs[idx, :len(boundaries)] = torch.LongTensor(boundaries)
        self.phn_seqs_len[idx] = len(phn_seq)
        self.clip_is[idx] = clip_i
        self.start_frames[idx] = start_frame
        self.end_frames[idx] = end_frame
        self.idx += 1

        if (self.idx == self.capacity):
            # The old factor of 1.1 needed ~40 full-array copies to reach a few million
            # rows, and int(capacity*1.1) == capacity for any capacity < 10, which would
            # silently stop growing and then index out of bounds on the next add.
            self._grow(max(self.capacity + 1, int(self.capacity * self.GROWTH_FACTOR)))

    def _grow(self, new_capacity):
        """Reallocate every parallel array to new_capacity, preserving rows [0, idx).

        Widths are read off the existing tensors rather than rebuilt by hand, so
        multi-column arrays keep their shape. bnd_seqs used to be reallocated here as
        1-D, which silently corrupted it on any growth.
        """
        def _resized(old, fill=None):
            new = torch.zeros((new_capacity,) + tuple(old.shape[1:]), dtype=old.dtype, device='cpu')
            if fill is not None:
                new.fill_(fill)
            new[:old.shape[0]] = old
            return new

        # One array at a time: each old buffer is released on rebind, so the transient
        # overhead is a single array rather than a second copy of the whole set.
        self.phn_seqs = _resized(self.phn_seqs, self._fill_value)
        self.grp_seqs = _resized(self.grp_seqs, self._fill_value)
        if (self.have_boundary_labels):
            self.bnd_seqs = _resized(self.bnd_seqs)
        self.phn_seqs_len = _resized(self.phn_seqs_len)
        self.clip_is = _resized(self.clip_is)
        self.start_frames = _resized(self.start_frames)
        self.end_frames = _resized(self.end_frames)
        self.capacity = new_capacity

    def __len__(self):
        return self.idx
    
    def get(self, idx):

        if (not self.have_boundary_labels):
            return self.phn_seqs[idx], self.grp_seqs[idx], self.phn_seqs_len[idx], self.clip_is[idx].item(), self.start_frames[idx], self.end_frames[idx], torch.tensor([-1], dtype=torch.long, device='cpu')
        else:
            return self.phn_seqs[idx], self.grp_seqs[idx], self.phn_seqs_len[idx], self.clip_is[idx].item(), self.start_frames[idx], self.end_frames[idx], self.bnd_seqs[idx]

    def clear(self):
        self.idx = 0
        del self.phn_seqs
        del self.grp_seqs
        del self.phn_seqs_len
    
    def finalize(self):
        # .clone() is required: a [:idx] slice is a view over the full init_capacity
        # storage, and torch.save serializes the underlying storage, not the view.
        # Without it the cache file holds the entire over-allocated arena on disk.
        self.phn_seqs = self.phn_seqs[:self.idx].clone()
        self.grp_seqs = self.grp_seqs[:self.idx].clone()
        self.phn_seqs_len = self.phn_seqs_len[:self.idx].clone()
        self.clip_is = self.clip_is[:self.idx].clone()
        self.start_frames = self.start_frames[:self.idx].clone()
        self.end_frames = self.end_frames[:self.idx].clone()
        self.capacity = self.idx

        if (self.have_boundary_labels):
            self.bnd_seqs = self.bnd_seqs[:self.idx].clone()
    



class Segments_Collator(Dataset):
    def __init__(self, paths_list, duration_max=10, ph_seq_max=64, regen_cache=False, resampler_sample_rate = 16000):
        print("Segments_Collator.", "Total clips:", len(paths_list))


        # Initialize basic parameters
        self.padding_ph_label = -100 # keep it -100 for standardization (same in datamanage.py) 2025-03-17
        self.ph_seq_min = 1
        self.ph_seq_max = ph_seq_max
        self.seg_duration_min = 0.05  # seconds, important to avoid invalid clips
        self.seg_duration_min_samples = int(self.seg_duration_min * resampler_sample_rate)
        self.seg_duration_max = duration_max  # seconds
        self.total_duration = 0
        self.resampler_sample_rate = resampler_sample_rate
        
        self.wav_len_max = int(self.seg_duration_max * self.resampler_sample_rate)
        
        # Initialize audio processing
        self.resampler = torchaudio.transforms.Resample(
            self.resampler_sample_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )

        
        if (paths_list[0]['ts'][0:2] == '$/'):  # when not an absolute path
            paths_list = [{**path, 'ts': os.path.join(METADATA_BASE, path['ts'][2:])} for path in paths_list]  
        if (os.path.exists(paths_list[0]['ts']) == False):
            raise Exception("File does not exist:", paths_list[0]['ts'], "Make sure the ts files exists.")
        first_segment_data = self.read_ts2(paths_list[0]['ts'])["segments"][0]
        
        
        # Set up disk-based caching for processed metadata
        cache_key = self._generate_cache_key(paths_list)
        self.cache_dir = os.path.join(tmp_dir, 'cache')
        print(self.cache_dir)
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.cache_file = os.path.join(self.cache_dir, f'collator_{cache_key}.pt')
        print("Cache file:", self.cache_file+'.txt')
        
        if (regen_cache) and (os.path.exists(self.cache_file)):
            confirm = input("Regenerate the collator cache? (y/n): ")
            if confirm.lower() != 'y':
                regen_cache = False
        
        if (not regen_cache):
            if self._try_load_from_cache():
                #print(f"Loaded metadata from cache")
                return
            
        self.all_clips_paths = {}
        self.unique_phones = set()
        
        self.have_boundary_labels = False
        if ('boundaries' in first_segment_data):
            #print("Boundaries found in the first file.")
            #exit()
            self.have_boundary_labels = True
        

        self.SegmentClass = SegmentClass(
            ph_seq_max=self.ph_seq_max,
            # Seed at one segment per clip -- the natural lower bound -- and let geometric
            # growth find the real size. The old *20 guess allocated 20x the rows this
            # dataset actually has (~63 GB of int64 for 3M segments) and, because
            # finalize() only sliced, wrote all of it to the cache file too.
            init_capacity=max(1024, len(paths_list)),
            fill_value=self.padding_ph_label,
            have_boundary_labels=self.have_boundary_labels
        )
        
        # Process the data
        self.new_collator(paths_list)
        self._save_to_cache()


    def new_collator(self, paths_list: List[Dict]):
        stats_collector = {
            'real_phn_seq_lens': deque(),
            'real_segment_durations': deque(),
            'real_word_counts': deque(),
            'total_words': 0
        }
        self.unique_phones = set()
        self.all_clips_paths = {}
        
        # Process files in parallel with memory-efficient batching
        num_workers = mp.cpu_count() - 1
        batch_size = 10000  # Process files in smaller batches to control memory
        
        def process_file_batch(batch_paths):
            batch_results = []
            for path_data in batch_paths:
                clip_i = path_data['index']
                self.all_clips_paths[clip_i] = path_data['audio']
                        
                segs_data = self.read_ts2(path_data['ts'])
                if (segs_data == None):
                    continue
                result = self._process_segments(segs_data, clip_i)
                if result:
                    batch_results.append(result)
            return batch_results
        
        # Add index to path_data
        for i, path_data in enumerate(paths_list):
            path_data['index'] = i
        
        # Process in batches
        with tqdm(total=len(paths_list), desc="Processing segments", leave=False) as pbar:
            for i in range(0, len(paths_list), batch_size):
                batch = paths_list[i:i + batch_size]
                
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(process_file_batch, [path_data]) 
                             for path_data in batch]
                    
                    for future in futures:
                        batch_results = future.result()
                        for file_results, file_stats in batch_results:
                            # Update statistics
                            stats_collector['real_phn_seq_lens'].extend(file_stats['ph_lens'])
                            stats_collector['real_segment_durations'].extend(file_stats['segment_durations'])
                            stats_collector['real_word_counts'].extend(file_stats['word_counts'])
                            stats_collector['total_words'] += file_stats['total']
                            
                            # Process results
                            for result in file_results:
                                self.SegmentClass.add(*result)
                                self.unique_phones.update(result[0])
                        
                        pbar.update(1)
                
                # Force garbage collection after each batch
                gc.collect()
        
        self.SegmentClass.finalize()
        self._print_statistics(
            list(stats_collector['real_phn_seq_lens']),
            list(stats_collector['real_segment_durations']),
            list(stats_collector['real_word_counts']),
            stats_collector['total_words']
        )

    def _process_segments(self, segs_data, clip_i):
        '''
        segs_data: list of segments
        example:
        {"segments": [{"start": 0.46, "end": 4.56, "text": " skinień skinień pani zawołałem nie przerywaj mi pan już", "ph": [0, 36, 32, 1, 54, 7, 54, 36, 32, 1, 54, 7, 54, 28, 18, 54, 1, 0, 37, 18, 44, 17, 61, 18, 61, 7, 52, 54, 7, 28, 40, 7, 59, 3, 44, 18, 60, 52, 1, 28, 18, 53, 60, 11, 40, 0], "phg": [0, 8, 6, 1, 12, 1, 12, 8, 6, 1, 12, 1, 12, 6, 4, 12, 1, 0, 9, 4, 9, 3, 15, 4, 15, 1, 12, 12, 1, 6, 8, 1, 14, 3, 9, 4, 15, 12, 1, 6, 4, 12, 15, 3, 8, 0], "word_num": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8, 8], "words": [" skinień", " skinień", " pani", " zawołałem", " nie", " przerywaj", " mi", " pan", " już"]}, {"start": 4.56, "end": 8.38, "text": " cię raz oto prosiłam odparła poczem tak dalej mówiła", "ph": [0, 38, 7, 59, 18, 36, 17, 30, 17, 28, 59, 17, 42, 1, 61, 18, 52, 17, 30, 28, 18, 59, 61, 18, 28, 17, 38, 7, 52, 30, 18, 32, 31, 18, 56, 7, 60, 52, 11, 44, 1, 61, 18, 0], "phg": [0, 10, 1, 14, 4, 8, 3, 6, 3, 6, 14, 3, 8, 1, 15, 4, 12, 3, 6, 6, 4, 14, 15, 4, 6, 3, 10, 1, 12, 6, 4, 6, 7, 4, 13, 1, 15, 12, 3, 9, 1, 15, 4, 0], "word_num": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8], "words": [" cię", " raz", " oto", " prosiłam", " odparła", " poczem", " tak", " dalej", " mówiła"]}, {"start": 8.9, "end": 10.66, "text": " zachciało mi się wystawiać cierpliwość", "ph": [0, 37, 18, 48, 38, 18, 61, 17, 52, 1, 42, 7, 44, 3, 36, 30, 18, 44, 18, 38, 38, 7, 59, 28, 56, 1, 44, 17, 42, 38, 0], "phg": [0, 9, 4, 8, 10, 4, 15, 3, 12, 1, 8, 1, 9, 3, 8, 6, 4, 9, 4, 10, 10, 1, 14, 6, 13, 1, 9, 3, 8, 10, 0], "word_num": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], "words": [" zachciało", " mi", " się", " wystawiać", " cierpliwość"]}, {"start": 10.66, "end": 12.76, "text": " moich kobiet na próby wszelkiego rodzaju", "ph": [0, 52, 17, 1, 48, 32, 17, 29, 60, 7, 30, 53, 18, 28, 59, 11, 29, 3, 43, 40, 7, 56, 32, 7, 33, 17, 59, 17, 39, 18, 60, 11, 0], "phg": [0, 12, 3, 1, 8, 6, 3, 7, 15, 1, 6, 12, 4, 6, 14, 3, 7, 3, 8, 8, 1, 13, 6, 1, 7, 3, 14, 3, 11, 4, 15, 3, 0], "word_num": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5], "words": [" moich", " kobiet", " na", " próby", " wszelkiego", " rodzaju"]}], "audio_path": "/path/to/audio/clip.flac"}
        
        '''
        results = []    # will hold the segment level sequences (previously word level)
        stats = {'ph_lens': [], 'segment_durations': [], 'word_counts':[], 'total': 0}
        
        
        
        for s in range(len(segs_data["segments"])):
            seg = segs_data["segments"][s]
            stats['total'] += 1
            ph_seq = seg['ph']
            grp_seq = seg['phg']
            
            if not ph_seq:  # Skip empty sequences
                continue
                
            ph_seq_len = len(ph_seq)
            if not (self.ph_seq_min <= ph_seq_len < self.ph_seq_max):
                continue
            
            seg_start = seg["start"]
            seg_end = seg["end"]
            
            
            seg_duration = seg_end - seg_start
            
            # Collect statistics
            stats['ph_lens'].append(ph_seq_len)
            stats['segment_durations'].append(seg_duration)
            stats['word_counts'].append(len(seg['words']))

            boundaries = None
            if ('boundaries' in seg):
                boundaries = seg['boundaries']
                
            # Add to results if meets criteria
            if (self.seg_duration_min <= seg_duration <= self.seg_duration_max):
                results.append((
                    ph_seq, 
                    grp_seq,
                    clip_i, 
                    int(seg_start * self.resampler_sample_rate), 
                    int(seg_end * self.resampler_sample_rate),
                    boundaries
                ))
                self.total_duration += seg_duration
        
        return results, stats


    def assure_total_classes(self, expected_classes):
        '''Check if the number of unique phonemes is as expected. Without the blank label.'''
        # sort unique_phones
        print("assure_total_classes, unique_phones:", list(self.unique_phones), "total:", len(self.unique_phones))
        #assert(len(self.unique_phones) == expected_classes)
        # at least half of the expected classes should be present, but not more than expected_classes
        if((len(self.unique_phones) >= (expected_classes/2)) and (len(self.unique_phones) <= expected_classes))==False:
            print("assure_total_classes, unique_phones:", list(self.unique_phones))
            confirm = input("Unique phonemes are less than expected. Continue? (y/n): ")
            if confirm.lower() != 'y':
                raise Exception("Unique phoneme classes are not as much as expected.", expected_classes)
        

    

    

    @torch.no_grad()
    def get_wav(self, clip_i, start_frame, end_frame):
        audio_path = self.all_clips_paths[clip_i]
        num_frames = end_frame - start_frame

        
        # Stream audio directly without caching
        try:
            wav, sr = torchaudio.load(
                audio_path, 
                frame_offset=start_frame, 
                num_frames=num_frames, 
                normalize=True,
                #ackend="ffmpeg" 
            )
        except Exception as e:
            raise Exception(f"Error loading audio file {audio_path}: {e}")
        
        wav = self.resampler(wav)
        assert wav.shape[1] <= num_frames
        if (wav.shape[1] < self.seg_duration_min_samples):
            raise Exception("Wav shape is too small:", wav.shape, audio_path, start_frame, end_frame)
        
        # Process wav
        wav = wav.mean(dim=0)
        wav = self._rms_normalize(wav)
        wav_len = wav.shape[0]
        # Pad or truncate
        if wav_len > self.wav_len_max:
            wav = wav[:self.wav_len_max]
            wav_len = wav.shape[0]
        else:
            wav = torch.nn.functional.pad(
                wav, 
                (0, self.wav_len_max - wav.shape[0]), 
                'constant', 
                0
            )
        
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
        ph_seq, grp_seq, ph_seq_len, clip_i, start_frame, end_frame, bnd_seq = self.SegmentClass.get(idx)
        wav, wav_len = self.get_wav(clip_i, start_frame, end_frame)
        #wav_len = wav.shape[0] # because of padding in get_wav, moved to get_wav
        wav = wav.unsqueeze(0)

        return ph_seq, grp_seq, wav, ph_seq_len, wav_len, bnd_seq
    

    def read_ts2(self, ts2_path):
        
        if (ts2_path[0:2] == '$/'): ts2_path = os.path.join(METADATA_BASE, ts2_path[2:])

        if (os.path.exists(ts2_path) == False):
            print("File does not exist:", ts2_path)
            return None
        try:
            with open(ts2_path, "r") as file:
                ts2_data = file.read()
                ts2_data = json.loads(ts2_data)
                return ts2_data
        except Exception as e:
            #print(f"Error reading TS2 file {ts2_path}: {e}")
            # remove the file if it is corrupted
            if os.path.exists(ts2_path):
                os.remove(ts2_path)
                print(f"Removed corrupted TS2 file: {ts2_path}")
            return None
    
    
    def _generate_cache_key(self, paths_list: List[Dict]) -> str:
        """Generate a unique cache key based on parameters and paths"""
        # Create a string containing all relevant parameters
        params_str = f"{self.ph_seq_min}_{self.ph_seq_max}_{self.seg_duration_min}_{self.seg_duration_max}"
        print("Params str:", params_str)
        # Add paths information (using just the first and last few paths to keep it manageable)
        paths_sample = paths_list[:3] + paths_list[-3:] if len(paths_list) > 6 else paths_list
        paths_str = "_".join(sorted(p['audio'] for p in paths_sample))
        
        # Create a hash of the combined string
        combined_str = f"{params_str}_{paths_str}_{len(paths_list)}"
        return hashlib.md5(combined_str.encode()).hexdigest()

    def _try_load_from_cache(self) -> bool:
        """Attempt to load the collator from cache"""
        if os.path.exists(self.cache_file):
            try:
                file_size_bytes = os.path.getsize(self.cache_file)
                file_size_gb = file_size_bytes / (1024 * 1024 * 1024)
                print(f"Loading cache file ({file_size_gb:.2f} GB): {self.cache_file}")
                cached_data = torch.load(self.cache_file, weights_only=False)
                
                # Verify cache validity by checking key attributes
                required_attrs = ['SegmentClass', 'all_clips_paths', 'unique_phones', 'wav_len_max']
                if all(hasattr(cached_data, attr) for attr in required_attrs):
                    # Load attributes from cached data
                    for key, value in cached_data.__dict__.items():
                        setattr(self, key, value)
                    return True
                
            except Exception as e:
                print(f"Cache loading failed: {e}")
                if os.path.exists(self.cache_file):
                    os.remove(self.cache_file)
        return False

    def _save_to_cache(self):
        """Save the current state to cache"""
        try:
            print(f"Saving collator to cache: {self.cache_file}")
            torch.save(self, self.cache_file)


            # Get file size in GB
            file_size_bytes = os.path.getsize(self.cache_file)
            file_size_gb = file_size_bytes / (1024 * 1024 * 1024)  # Convert bytes to GB

            
            print(f"Cache file size: {file_size_gb:.2f} GB")
        except Exception as e:
            print(f"Cache saving failed: {e}")
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)

    def _print_statistics(self, real_phn_seq_lens, real_seg_duration, real_word_counts, total_words):
        """Print statistical summaries of the processed data"""
        # Convert to tensors for efficient computation
        phn_tensor = torch.tensor(real_phn_seq_lens, dtype=torch.float)
        durations_tensor = torch.tensor(real_seg_duration, dtype=torch.float)
        word_count_tensor = torch.tensor(real_word_counts, dtype=torch.float)   
        if len(phn_tensor) == 0:
            raise Exception("No valid words found. Make sure the ts files exists.")
            
        stats_output = []

        stats_output.append(
            "phonemes-per-seq stats min, median, mean, max:\t"
            f"{phn_tensor.min().item():.3f}\t"
            f"{torch.median(phn_tensor).item():.3f}\t"
            f"{phn_tensor.mean().item():.3f}\t"
            f"{phn_tensor.max().item():.3f}\t std:"
            f"{phn_tensor.std().item():.3f}"
        )

        stats_output.append(
            "duration-s stats min, median, mean, max:\t"
            f"{durations_tensor.min().item():.3f}\t"
            f"{torch.median(durations_tensor).item():.3f}\t"
            f"{durations_tensor.mean().item():.3f}\t"
            f"{durations_tensor.max().item():.3f}\t std:"
            f"{durations_tensor.std().item():.3f}"
        )

        stats_output.append(
            "words-per-seg stats min, median, mean, max:\t"
            f"{word_count_tensor.min().item():.3f}\t"
            f"{torch.median(word_count_tensor).item():.3f}\t"
            f"{word_count_tensor.mean().item():.3f}\t"
            f"{word_count_tensor.max().item():.3f}\t std:"
            f"{word_count_tensor.std().item():.3f}"
        )

        

        filtered_ratio = len(self.SegmentClass) / total_words
        stats_output.append(f"Unique phonemes: {len(self.unique_phones)}")
        stats_output.append(
            f"Total segments: {total_words} "
            f"Filtered segments: {len(self.SegmentClass)} "
            f"{(filtered_ratio * 100):.3f}% "
            f"Total hours: {self.total_duration / 3600:.3f}"
        )

        # Print the statistics
        for line in stats_output:
            print(line)

        if (filtered_ratio < 0.75):
            confirm = input("Filteration reduced the total samples to less than 75%. Continue? (y/n): ")
            if confirm.lower() != 'y':
                raise Exception("Filtered ratio is not in the expected range.")

        # Write the statistics to a text file if text_save_path is provided
        if self.cache_file is not None:
            with open(self.cache_file +".txt", "w") as file:
                for line in stats_output:
                    file.write(line + "\n")
                
        # Free memory
        del phn_tensor, durations_tensor, word_count_tensor
        gc.collect()

def task_CHECK_data_collator(paths_list):

    #paths_list = libri.assort_parts("dev-clean")

    C = SegmentClass(paths_list)

    print("Total words:", len(C))
    print("Total clips:", len(paths_list))
    print("Unique phonemes:", len(C.unique_phones))
    print("First sample:",C[0][0], C[0][1].shape[0]/16000)