# scripts/predict.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
from tensorflow import keras
from src.feature_extractor import CombinedFeatureExtractor, CNN_FEATURE_DIM
from config import SELECTED_CLASSES, MAX_SEQ_LENGTH

def predict_video(model_path, video_path):
    """
    Predict action class for a single video file.

    Args:
        model_path (str): Path to the trained model (.h5 file)
        video_path (str): Path to the video file

    Returns:
        tuple: (predicted_class_name, confidence_score)
    """
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    model = keras.models.load_model(model_path)
    print(f"Loaded model: {model_path}")

    # Extract features from video
    extractor = CombinedFeatureExtractor()
    print(f"Processing video: {video_path}")

    # Get CNN features (skeleton features not available for prediction)
    cnn_features = extractor.cnn_extract_from_video(video_path)

    # For prediction, we only have CNN features, so skeleton features are zeros
    sk_dim = 17 * 3  # 17 keypoints * (x, y, score)
    skeleton_features = np.zeros((MAX_SEQ_LENGTH, sk_dim), dtype=np.float32)

    # Combine features
    features = np.concatenate([cnn_features, skeleton_features], axis=-1)
    features = features.reshape(1, MAX_SEQ_LENGTH, -1)  # Add batch dimension

    # Create mask: frames with non-zero CNN features are valid
    mask = np.any(cnn_features != 0.0, axis=-1).reshape(1, MAX_SEQ_LENGTH)

    # Make prediction
    predictions = model.predict([features, mask], verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Map class index to name if we have the mapping, otherwise use index
    if predicted_class_idx < len(SELECTED_CLASSES):
        predicted_class = SELECTED_CLASSES[predicted_class_idx]
    else:
        predicted_class = f"Class_{predicted_class_idx}"

    return predicted_class, confidence

def main():
    parser = argparse.ArgumentParser(description='Predict action from video using trained model')
    parser.add_argument('model_path', help='Path to trained model (.h5 file)')
    parser.add_argument('video_path', help='Path to video file to predict')

    args = parser.parse_args()

    try:
        predicted_class, confidence = predict_video(args.model_path, args.video_path)

        print("\nPrediction Results:")
        print(f"   Video: {os.path.basename(args.video_path)}")
        print(f"   Predicted Action: {predicted_class}")
        print(f"   Confidence: {confidence:.2%}")

        # Show top 3 predictions
        print("\nTop 3 Predictions:")
        # Get predictions and sort
        model = keras.models.load_model(args.model_path)
        extractor = CombinedFeatureExtractor()

        cnn_features = extractor.cnn_extract_from_video(args.video_path)
        sk_dim = 17 * 3
        skeleton_features = np.zeros((MAX_SEQ_LENGTH, sk_dim), dtype=np.float32)
        features = np.concatenate([cnn_features, skeleton_features], axis=-1).reshape(1, MAX_SEQ_LENGTH, -1)
        mask = np.any(cnn_features != 0.0, axis=-1).reshape(1, MAX_SEQ_LENGTH)

        predictions = model.predict([features, mask], verbose=0)[0]

        # Get top 3 indices
        top_3_indices = np.argsort(predictions)[-3:][::-1]

        for i, idx in enumerate(top_3_indices, 1):
            if idx < len(SELECTED_CLASSES):
                class_name = SELECTED_CLASSES[idx]
            else:
                class_name = f"Class_{idx}"
            print(f"   {i}. {class_name}: {predictions[idx]:.2%}")

    except FileNotFoundError as e:
        print(f"{e}")
        print("Make sure the model file exists and the video path is correct")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
