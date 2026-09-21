import lightning as L
from torch.utils.data import DataLoader, Subset
from p3cupe.SegmentsCollator import Segments_Collator
import gc
from typing import List, Optional, Union
from contextlib import contextmanager

@contextmanager
def clean_memory():
    try:
        yield
    finally:
        gc.collect()


def evenly_spaced_indices(total: int, keep: int):
    '''
    Deterministically pick `keep` of `total` indices, spread evenly over the ordering.

    Used to shrink the validation set: the full libri dev-clean split is ~2700 segments, and
    validation cost is dominated by two PER-CLIP Python loops - `process_alignments` runs a
    Viterbi per clip, and `calculate_per` decodes per clip - so wall-clock falls roughly
    linearly with this.

    Evenly spaced rather than randomly sampled, deliberately:
      - it needs no seed, so it is identical across runs, across resumes, and across DDP ranks
        without any coordination;
      - path lists are ordered by speaker, so a stride keeps speaker coverage uniform, whereas
        a random draw of a few hundred can over- or under-represent speakers;
      - the indices stay sorted, so audio is still read roughly in disk order.

    The subset is fixed, so validation metrics use the same clips each epoch.
    '''
    total = int(total)
    keep = int(keep)
    if keep <= 0 or keep >= total:
        return list(range(total))
    stride = total / keep
    return sorted({min(int(i * stride), total - 1) for i in range(keep)})

class MyDataModule(L.LightningDataModule):
    def __init__(self, expected_phoneme_classes: Optional[int] = None, 
                 num_workers: int = 10, 
                 batch_size: int = 32,
                 persistent_workers: bool = True):
        super().__init__()
        self.expected_phoneme_classes = expected_phoneme_classes
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.persistent_workers = persistent_workers
        self.ready_state = 'unprepared'
        
        # Initialize as None to prevent memory leaks
        self.paths_list = None
        self.train_paths = None
        self.val_paths = None
        self.test_paths = None
        self.train_collator = None
        self.val_collator = None
        self.test_collator = None
        self.val_dataset = None      # val_collator, or a fixed Subset of it
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
    
    def load_paths_split(self, paths_list: List, split_at: List[float] = [0.85, 0.90]):
        with clean_memory():
            total_len = len(paths_list)
            split_lens = [
                int(total_len * split_at[0]),
                int(total_len * split_at[1]) - int(total_len * split_at[0]),
                total_len - int(total_len * split_at[1])
            ]
            print("prepare_data splits:", split_lens)
            
            # Avoid storing the full paths_list
            self.train_paths = paths_list[:split_lens[0]]
            self.val_paths = paths_list[split_lens[0]:split_lens[0]+split_lens[1]]
            self.test_paths = paths_list[split_lens[0]+split_lens[1]:]
            
            # Clear the original paths_list
            self.paths_list = None
            del paths_list
            gc.collect()

    def load_paths_parts(self, train_paths: List = [], 
                        val_paths: List = [], 
                        test_paths: List = []):
        with clean_memory():
            self.train_paths = train_paths
            self.val_paths = val_paths
            self.test_paths = test_paths
            split_lens = [len(self.train_paths), len(self.val_paths), len(self.test_paths)]
            print("prepare_data splits:", split_lens)

    def _clear_unused_data(self, stage: str):
        """Clear data that's not needed for the current stage"""
        if stage == 'fit':
            self.test_collator = None
            self.test_loader = None
        elif stage == 'test':
            self.train_collator = None
            self.val_collator = None
            self.val_dataset = None
            self.train_loader = None
            self.valid_loader = None
        gc.collect()

    def setup(self, stage: Optional[str] = None, duration_max=10, ph_seq_max=64, resampler_sample_rate: Optional[Union[int, float]] = 16000, regen_cache: bool = False, val_max_segments: int = 0):
        if self.ready_state == stage:
            print("Data already ready.")
            return
            
        with clean_memory():
            if stage == 'fit':
                print(".............................train:")
                self.train_collator = Segments_Collator(self.train_paths, duration_max=duration_max, ph_seq_max=ph_seq_max, regen_cache=regen_cache, resampler_sample_rate=resampler_sample_rate)
                print(".............................val:")
                self.val_collator = Segments_Collator(self.val_paths, duration_max=duration_max, ph_seq_max=ph_seq_max, regen_cache=regen_cache, resampler_sample_rate=resampler_sample_rate)

                if self.expected_phoneme_classes is not None:
                    self.train_collator.assure_total_classes(self.expected_phoneme_classes)
                    self.val_collator.assure_total_classes(self.expected_phoneme_classes)
                
                first_ph_seq = self.train_collator[0][0]
                print("First phoneme sequence:", first_ph_seq)

                self.train_total = len(self.train_collator)
                self.ph_seq_max = self.train_collator.ph_seq_max
                self.wav_len_max = self.train_collator.wav_len_max
                # Taken from the FULL collator on purpose: it sets the padding width, so a
                # subset must not be allowed to shrink it below what any clip needs.
                self.wav_len_max_val = self.val_collator.wav_len_max

                # Optional deterministic subsample of the validation set. `val_total` is what
                # `model.configure()` uses to preallocate the metric buffers, so this shrinks
                # those too, not just the number of forward passes.
                full_val_total = len(self.val_collator)
                if val_max_segments and 0 < val_max_segments < full_val_total:
                    indices = evenly_spaced_indices(full_val_total, val_max_segments)
                    self.val_dataset = Subset(self.val_collator, indices)
                    print(f"val subsample: {len(indices)} of {full_val_total} segments "
                          f"(every ~{full_val_total / len(indices):.1f}th, fixed - identical "
                          f"across runs, resumes and DDP ranks)")
                else:
                    self.val_dataset = self.val_collator
                self.val_total = len(self.val_dataset)

            elif stage == 'test' or stage is None:
                print(".............................test:")
                self.test_collator = Segments_Collator(self.test_paths, duration_max=duration_max, ph_seq_max=ph_seq_max, regen_cache=regen_cache, resampler_sample_rate=resampler_sample_rate)
                
                if self.expected_phoneme_classes is not None:
                    self.test_collator.assure_total_classes(self.expected_phoneme_classes)
                
                self.test_total = len(self.test_collator)
                self.ph_seq_max = self.test_collator.ph_seq_max
                self.wav_len_max = self.test_collator.wav_len_max
            else:
                raise ValueError("Stage must be 'fit' or 'test'")

            self.ready_state = stage
            self._clear_unused_data(stage)
            print(".............................data ready.")

    def _create_dataloader(self, dataset, shuffle: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=shuffle,  # Only drop last for training
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True
        )

    def train_dataloader(self):
        self.train_loader = self._create_dataloader(self.train_collator, shuffle=True)
        return self.train_loader

    def val_dataloader(self):
        # `val_dataset` is the collator itself, or a fixed Subset of it when `val_max_segments`
        # is set. `shuffle=False`, so the same segments are scored in the same order every epoch.
        self.valid_loader = self._create_dataloader(
            getattr(self, 'val_dataset', None) or self.val_collator, shuffle=False)
        return self.valid_loader

    def test_dataloader(self):
        self.test_loader = self._create_dataloader(self.test_collator, shuffle=False)
        return self.test_loader
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after training/testing is done"""
        with clean_memory():
            if stage == 'fit':
                self.train_collator = None
                self.val_collator = None
                self.val_dataset = None
                self.train_loader = None
                self.valid_loader = None
            elif stage == 'test':
                self.test_collator = None
                self.test_loader = None
            gc.collect()