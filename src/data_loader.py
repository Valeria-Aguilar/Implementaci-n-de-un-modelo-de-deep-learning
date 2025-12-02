# src/data_loader.py
import pickle
import os

class UCF101PickleIndex:
    """
    Loads the pickle index (split + annotations list) and provides access.
    Does NOT assume presence of raw videos.
    """
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)
        self.split = self.data.get('split', {})
        self.annotations_list = self.data.get('annotations', [])
        # build dict keyed by frame_dir (e.g., 'v_ApplyLipstick_g25_c02')
        self.ann_by_frame = {}
        for entry in self.annotations_list:
            key = entry.get('frame_dir')  # matches split entries
            if key:
                self.ann_by_frame[key] = entry

    def get_split_list(self, split_name):
        """Return list of frame_dir names under split_name (e.g., 'train1')"""
        return self.split.get(split_name, [])
