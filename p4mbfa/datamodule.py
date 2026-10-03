import gc
from contextlib import contextmanager
from typing import Dict, List, Optional

import lightning as L
from torch.utils.data import DataLoader, Subset

from p4mbfa.SegmentsCollator import Segments_Collator


@contextmanager
def clean_memory():
    try:
        yield
    finally:
        gc.collect()


def evenly_spaced_indices(total: int, keep: int):
    '''
    Deterministically pick `keep` of `total` indices, spread evenly over the ordering. Needs no
    seed, so it is identical across runs, resumes and DDP ranks; path lists are ordered by
    language, so a stride keeps every language represented.
    '''
    total = int(total)
    keep = int(keep)
    if keep <= 0 or keep >= total:
        return list(range(total))
    stride = total / keep
    return sorted({min(int(i * stride), total - 1) for i in range(keep)})





def assert_disjoint(train_paths: List, val_paths: List):
    overlap = {p['audio'] for p in train_paths} & {p['audio'] for p in val_paths}
    if overlap:
        raise ValueError(f"{len(overlap)} audio files are in BOTH train and val (e.g. "
                         f"{sorted(overlap)[0]}). splits.get_splits returns the same list for "
                         f"both on some sources (fleurs does).")


class MyDataModule(L.LightningDataModule):
    def __init__(self, collator_config: Dict, num_workers: int = 10, batch_size: int = 32,
                 persistent_workers: bool = True):
        super().__init__()
        self.collator_config = dict(collator_config)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.persistent_workers = persistent_workers and num_workers > 0
        self.ready_state = 'unprepared'

        self.train_paths = None
        self.val_paths = None
        self.train_collator = None
        self.val_collator = None
        self.val_dataset = None      # val_collator, or a fixed Subset of it
        self.train_loader = None
        self.valid_loader = None

    def load_paths_parts(self, train_paths: List, val_paths: List):
        if not train_paths or not val_paths:
            raise ValueError(f"need non-empty train and val paths, got {len(train_paths or [])} "
                             f"and {len(val_paths or [])}")
        assert_disjoint(train_paths, val_paths)
        self.train_paths = train_paths
        self.val_paths = val_paths
        print("prepare_data splits (files):", [len(train_paths), len(val_paths)])

    def setup(self, stage: Optional[str] = None, regen_cache: bool = False,
              val_max_segments: int = 0):
        if self.ready_state == stage:
            print("Data already ready.")
            return
        if stage != 'fit':
            raise ValueError(f"only stage 'fit' is supported, got {stage!r}")

        with clean_memory():
            print(".............................train:")
            self.train_collator = Segments_Collator(self.train_paths, name='train',
                                                    regen_cache=regen_cache,
                                                    **self.collator_config)
            print(".............................val:")
            self.val_collator = Segments_Collator(self.val_paths, name='val',
                                                  regen_cache=regen_cache,
                                                  **self.collator_config)

            first = self.train_collator[0][0]
            print("First phoneme sequence:", self.train_collator.decode(first))

            self.train_total = len(self.train_collator)
            self.ph_seq_max = self.train_collator.ph_seq_max
            self.wav_len_max = self.train_collator.wav_len_max
            self.wav_len_max_val = self.val_collator.wav_len_max

            full_val_total = len(self.val_collator)
            if val_max_segments and 0 < val_max_segments < full_val_total:
                indices = evenly_spaced_indices(full_val_total, val_max_segments)
                self.val_dataset = Subset(self.val_collator, indices)
                print(f"val subsample: {len(indices)} of {full_val_total} segments "
                      f"(fixed - identical across runs, resumes and DDP ranks)")
            else:
                self.val_dataset = self.val_collator
            self.val_total = len(self.val_dataset)

            self.ready_state = stage
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
        self.valid_loader = self._create_dataloader(self.val_dataset, shuffle=False)
        return self.valid_loader

    def teardown(self, stage: Optional[str] = None):
        with clean_memory():
            if stage == 'fit':
                self.train_collator = None
                self.val_collator = None
                self.val_dataset = None
                self.train_loader = None
                self.valid_loader = None
