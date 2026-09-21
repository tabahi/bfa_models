

from typing import List, Tuple, Optional
from contextlib import contextmanager
import gc

import paths_def

@contextmanager
def memory_cleanup():
    """Context manager to handle memory cleanup after operations"""
    try:
        yield
    finally:
        gc.collect()

def merge_path_lists(list1: Tuple[List, float], 
                    list2: Tuple[List, float]) -> Tuple[List, float]:
    """
    Efficiently merge two path lists and their hours while managing memory
    
    Args:
        list1: Tuple of (paths_list, total_hours) for first dataset
        list2: Tuple of (paths_list, total_hours) for second dataset
        
    Returns:
        Tuple of (merged_paths_list, combined_hours)
    """
    paths1, hours1 = list1
    paths2, hours2 = list2
    
    with memory_cleanup():
        # Extend in place to avoid creating temporary copies
        paths1.extend(paths2)
        combined_hours = hours1 + hours2
        
    return paths1, combined_hours



def get_splits(data_src: str,
                  exclude_langs: Optional[List[str]] = None,
                  include_langs: Optional[List[str]] = None, dev_mode=False) -> Tuple[List, float, Optional[Tuple[List, float]]]:
    """
    Get training and validation paths with efficient memory handling
    
    Args:
        data_src: Source dataset identifier
        exclude_langs: Languages to exclude
        include_langs: Languages to include
        
    Returns:
        Tuple of (train_paths, train_hours, optional validation_data)
    """
    
    
    if data_src == "LibriTTS_dev":
        with memory_cleanup():
            train_data = paths_def.get_assorted_paths("LibriTTS_dev_clean", limit_clips=100 if dev_mode else None)
            val_data = paths_def.get_assorted_paths("LibriTTS_dev_clean", limit_clips=100 if dev_mode else None)
            
            return train_data[0], train_data[1], val_data

    elif data_src == "LibriTTS_100h":
        with memory_cleanup():
            train_data = paths_def.get_assorted_paths("LibriTTS_train_100", limit_clips=100 if dev_mode else None)
            val_data = paths_def.get_assorted_paths("LibriTTS_dev_clean", limit_clips=100 if dev_mode else None)
            
            return train_data[0], train_data[1], val_data
    elif data_src == "LibriTTS_360h":
        with memory_cleanup():
            train_data = paths_def.get_assorted_paths("LibriTTS_train_360", limit_clips=100 if dev_mode else None)
            val_data = paths_def.get_assorted_paths("LibriTTS_dev_clean", limit_clips=100 if dev_mode else None)
            
            return train_data[0], train_data[1], val_data
    elif data_src == "libri_dev_clean": # for debugging only
        with memory_cleanup():
            train_data = paths_def.get_assorted_paths("libri_dev_clean", limit_clips=100 if dev_mode else None)
            val_data = paths_def.get_assorted_paths("libri_dev_clean", limit_clips=100 if dev_mode else None)
            return train_data[0], train_data[1], val_data
    else:
        train_data = paths_def.get_assorted_paths(data_src, None)

        return train_data[0], train_data[1], None