# scripts/evaluate_model.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import SELECTED_CLASSES

DATA_DIR = "data/processed"

def load_and_filter_data():
    """Load data and filter to only selected classes"""
    X_test = np.load(os.path.join(DATA_DIR, "test_features.npy"))
    M_test = np.load(os.path.join(DATA_DIR, "test_masks.npy"))
    Y_test = np.load(os.path.join(DATA_DIR, "test_labels.npy"))

    # Filter data to only include selected classes
    selected_classes = list(range(len(SELECTED_CLASSES)))  # [0, 1, 2, 3, 4]

    # Filter test data
    test_mask = np.isin(Y_test, selected_classes)
    X_test = X_test[test_mask]
    M_test = M_test[test_mask]
    Y_test = Y_test[test_mask]

    # Create label mapping: original label -> new label (0 to len(selected_classes)-1)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(selected_classes)}

    # Remap labels
    Y_test = np.array([label_mapping[label] for label in Y_test])

    # ensure labels shape (N,1)
    if Y_test.ndim == 1:
        Y_test = Y_test.reshape(-1, 1)

    return (X_test, M_test), Y_test

def evaluate_model(model_path):
    """Evaluate model on filtered test data"""
    print(f"Loading model: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    model = keras.models.load_model(model_path)
    print("Model loaded successfully")
    print(f"   Expected classes: {model.output_shape[-1]}")
    print(f"   Selected classes: {SELECTED_CLASSES}")

    # Load and filter test data
    (X_test, M_test), Y_test = load_and_filter_data()
    print("\nTest data (filtered):")
    print(f"   Shape: {X_test.shape}")
    print(f"   Classes: {len(SELECTED_CLASSES)}")

    # Show class distribution
    unique, counts = np.unique(Y_test.flatten(), return_counts=True)
    for i, (class_idx, count) in enumerate(zip(unique, counts)):
        print(f"   {SELECTED_CLASSES[class_idx]}: {count} samples")

    # Evaluate
    print("\nEvaluating...")
    loss, acc = model.evaluate([X_test, M_test], Y_test, verbose=1)

    print("\nResults:")
    print(f"   Test Loss: {loss:.4f}")
    print(f"   Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Per-class accuracy
    predictions = model.predict([X_test, M_test], verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = Y_test.flatten()

    print("\nPer-class accuracy:")
    for i, class_name in enumerate(SELECTED_CLASSES):
        class_mask = (true_classes == i)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(pred_classes[class_mask] == i)
            print(f"   {class_name}: {class_acc:.2%}")
        else:
            print(f"   {class_name}: No samples")

    return acc

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on selected classes')
    parser.add_argument('model_path', help='Path to trained model (.h5 file)')

    args = parser.parse_args()

    try:
        accuracy = evaluate_model(args.model_path)
        print(f"\nFinal Accuracy: {accuracy:.4f}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
