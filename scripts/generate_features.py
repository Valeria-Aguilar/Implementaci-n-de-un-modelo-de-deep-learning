# scripts/generate_features.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from tqdm import tqdm
from src.data_loader import UCF101PickleIndex
from src.feature_extractor import CombinedFeatureExtractor, CNN_FEATURE_DIM
from config import MAX_SEQ_LENGTH

# CONFIG: update paths if needed
PICKLE_PATH = "data/ucf101_2d.pkl"   # your pickle file
VIDEO_ROOT = "data/videos"           # optional folder with raw videos named like frame_dir + .avi
OUT_DIR = "data/processed"
USE_SCORE = True                     # use keypoint scores (True -> skeleton_dim=51)
os.makedirs(OUT_DIR, exist_ok=True)

def find_video_file(frame_dir):
    # try common extensions
    for ext in (".avi", ".mp4", ".mpg", ".mpeg"):
        p = os.path.join(VIDEO_ROOT, frame_dir + ext)
        if os.path.exists(p):
            return p
    # sometimes videos may be nested, attempt shallow search
    for root, _, files in os.walk(VIDEO_ROOT):
        for f in files:
            if f.startswith(frame_dir) and f.lower().endswith((".avi", ".mp4")):
                return os.path.join(root, f)
    return None

def main():
    index = UCF101PickleIndex(PICKLE_PATH)
    extractor = CombinedFeatureExtractor()

    # compute skeleton feature dim
    sk_dim = 17 * (3 if USE_SCORE else 2)
    total_dim = CNN_FEATURE_DIM + sk_dim

    def process_split(split_name):
        ids = index.get_split_list(split_name)
        N = len(ids)
        print(f"[INFO] Processing split {split_name}, {N} samples")
        X = np.zeros((N, MAX_SEQ_LENGTH, total_dim), dtype=np.float32)
        M = np.zeros((N, MAX_SEQ_LENGTH), dtype=bool)
        Y = np.zeros((N,), dtype=np.int32)

        for i, frame_dir in enumerate(tqdm(ids, desc=split_name)):
            ann = index.ann_by_frame.get(frame_dir)
            if ann is None:
                # missing annotation -> skip as zeros
                continue

            # skeleton per-frame features (MAX_SEQ_LENGTH, sk_dim)
            sk_feats = extractor.skeleton_features_from_annotation(ann, use_score=USE_SCORE)  # (MAX_SEQ_LENGTH, sk_dim)

            # CNN features from video if available, else zeros
            vid_path = find_video_file(frame_dir)
            if vid_path:
                cnn_feats = extractor.cnn_extract_from_video(vid_path)
            else:
                cnn_feats = np.zeros((MAX_SEQ_LENGTH, CNN_FEATURE_DIM), dtype=np.float32)

            # concatenate per frame
            combined = np.concatenate([cnn_feats, sk_feats], axis=-1)  # (MAX_SEQ_LENGTH, total_dim)
            X[i] = combined

            # mask: frames with nonzero skeleton coords (or nonzero cnn) mark as valid
            mask = np.any(sk_feats != 0.0, axis=-1) | np.any(cnn_feats != 0.0, axis=-1)
            M[i] = mask
            Y[i] = int(ann.get('label', -1))

        return X, M, Y

    # train/test splits: try common keys train1/test1 or train/test
    train_key = "train1" if "train1" in index.split else "train"
    test_key = "test1" if "test1" in index.split else "test"

    X_train, M_train, Y_train = process_split(train_key)
    X_test, M_test, Y_test = process_split(test_key)

    # save
    np.save(os.path.join(OUT_DIR, "train_features.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "train_masks.npy"), M_train)
    np.save(os.path.join(OUT_DIR, "train_labels.npy"), Y_train)

    np.save(os.path.join(OUT_DIR, "test_features.npy"), X_test)
    np.save(os.path.join(OUT_DIR, "test_masks.npy"), M_test)
    np.save(os.path.join(OUT_DIR, "test_labels.npy"), Y_test)

    print("[DONE] saved processed arrays to", OUT_DIR)
    print("Shapes:", X_train.shape, M_train.shape, Y_train.shape)

if __name__ == "__main__":
    main()
