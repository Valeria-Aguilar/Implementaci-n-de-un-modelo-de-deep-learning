# src/feature_extractor.py
import os
import numpy as np
from tensorflow import keras
import cv2
from config import IMG_SIZE, MAX_SEQ_LENGTH

# CNN feature dim for InceptionV3 pooling="avg"
CNN_FEATURE_DIM = 2048

class CombinedFeatureExtractor:
    """
    Provides:
      - cnn_extract_from_video(path) -> (MAX_SEQ_LENGTH, CNN_FEATURE_DIM)
      - skeleton_features_from_annotation(annotation, use_score=True) -> (MAX_SEQ_LENGTH, SKELETON_DIM)
    """

    def __init__(self, use_gpu=True):
        # Build CNN model (InceptionV3)
        # Note: loading weights may be slower on CPU. GPU recommended.
        self.cnn = keras.applications.InceptionV3(
            weights="imagenet", include_top=False, pooling="avg",
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
        preprocessed = keras.applications.inception_v3.preprocess_input(inputs)
        outputs = self.cnn(preprocessed)
        self.cnn_model = keras.Model(inputs, outputs, name="inceptionv3_features")

    # ---------- CNN helpers ----------
    def load_video_frames(self, path, max_frames=MAX_SEQ_LENGTH):
        """Return numpy array of frames (N, H, W, 3) in RGB. If cannot open, return empty array."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return np.array([])

        frames = []
        def crop_center_square(frame):
            h, w = frame.shape[:2]
            m = min(h, w)
            sx = (w - m) // 2
            sy = (h - m) // 2
            return frame[sy:sy+m, sx:sx+m]

        try:
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()

        if len(frames) == 0:
            return np.array([])

        return np.stack(frames, axis=0)

    def cnn_extract_from_video(self, video_path):
        """
        Returns array shape (MAX_SEQ_LENGTH, CNN_FEATURE_DIM).
        If video cannot be read, returns zeros.
        """
        frames = self.load_video_frames(video_path, max_frames=MAX_SEQ_LENGTH)
        if frames.size == 0:
            return np.zeros((MAX_SEQ_LENGTH, CNN_FEATURE_DIM), dtype=np.float32)

        # Predict batch wise (may be large but MAX_SEQ_LENGTH typically small)
        feats = self.cnn_model.predict(frames, verbose=0)  # (num_frames, 2048)
        # pad / truncate
        if feats.shape[0] < MAX_SEQ_LENGTH:
            pad = np.zeros((MAX_SEQ_LENGTH - feats.shape[0], CNN_FEATURE_DIM), dtype=feats.dtype)
            feats = np.vstack([feats, pad])
        else:
            feats = feats[:MAX_SEQ_LENGTH]
        return feats.astype(np.float32)

    # ---------- Skeleton helpers ----------
    def skeleton_features_from_annotation(self, ann, use_score=False):
        # Extract keypoints and scores from the annotation structure
        keypoints_array = ann["keypoint"][0]  # Shape: (frames, 17, 2)
        scores_array = ann.get("keypoint_score", np.ones_like(keypoints_array[..., 0]))[0]  # Shape: (frames, 17)

        seq = []

        for frame_idx in range(keypoints_array.shape[0]):
            keypoints = keypoints_array[frame_idx]  # Shape: (17, 2)
            scores = scores_array[frame_idx] if use_score else np.ones(17)

            if len(keypoints) != 17:
                continue

            if use_score:
                flat = []
                for kp_idx in range(17):
                    flat += [keypoints[kp_idx][0], keypoints[kp_idx][1], scores[kp_idx]]
            else:
                flat = []
                for kp_idx in range(17):
                    flat += [keypoints[kp_idx][0], keypoints[kp_idx][1]]

            seq.append(flat)

        seq = np.array(seq, dtype=np.float32)

        # Pad/truncate
        if len(seq) < MAX_SEQ_LENGTH:
            pad = np.zeros((MAX_SEQ_LENGTH - len(seq), seq.shape[1]), dtype=np.float32)
            seq = np.vstack([seq, pad])
        else:
            seq = seq[:MAX_SEQ_LENGTH]

        return seq
