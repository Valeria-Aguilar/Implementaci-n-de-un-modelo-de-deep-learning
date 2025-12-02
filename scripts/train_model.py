# scripts/train_model.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
from tensorflow import keras
from models.baseline_model import BaselineModel
from models.rnn_model import RNNModel as OptimizedModel
from models.improved_model import ImprovedModel
from config import SELECTED_CLASSES, MAX_SEQ_LENGTH

DATA_DIR = "data/processed"

def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "train_features.npy"))
    M_train = np.load(os.path.join(DATA_DIR, "train_masks.npy"))
    Y_train = np.load(os.path.join(DATA_DIR, "train_labels.npy"))

    X_test = np.load(os.path.join(DATA_DIR, "test_features.npy"))
    M_test = np.load(os.path.join(DATA_DIR, "test_masks.npy"))
    Y_test = np.load(os.path.join(DATA_DIR, "test_labels.npy"))

    # Filter data to only include selected classes
    selected_classes = list(range(len(SELECTED_CLASSES)))  # [0, 1, 2, 3, 4]

    # Filter training data
    train_mask = np.isin(Y_train, selected_classes)
    X_train = X_train[train_mask]
    M_train = M_train[train_mask]
    Y_train = Y_train[train_mask]

    # Filter test data
    test_mask = np.isin(Y_test, selected_classes)
    X_test = X_test[test_mask]
    M_test = M_test[test_mask]
    Y_test = Y_test[test_mask]

    # Create label mapping: original label -> new label (0 to len(selected_classes)-1)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(selected_classes)}

    # Remap labels
    Y_train = np.array([label_mapping[label] for label in Y_train])
    Y_test = np.array([label_mapping[label] for label in Y_test])

    # ensure labels shape (N,1)
    if Y_train.ndim == 1:
        Y_train = Y_train.reshape(-1, 1)
    if Y_test.ndim == 1:
        Y_test = Y_test.reshape(-1, 1)

    print("Loaded shapes:", X_train.shape, M_train.shape, Y_train.shape)
    print(f"Selected classes: {SELECTED_CLASSES}")
    print(f"Training samples per class: {np.bincount(Y_train.flatten())}")
    print(f"Test samples per class: {np.bincount(Y_test.flatten())}")
    return (X_train, M_train), Y_train, (X_test, M_test), Y_test

def build_model(kind, num_classes):
    if kind == "baseline":
        return BaselineModel(num_classes).build_model()
    elif kind == "optimized":
        return OptimizedModel(num_classes).build_model()
    elif kind == "improved":
        return ImprovedModel(num_classes).build_model()
    else:
        raise ValueError("Unknown model type")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "optimized", "improved"], default="improved")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    (X_train, M_train), Y_train, (X_test, M_test), Y_test = load_data()
    num_classes = len(SELECTED_CLASSES)  # Use selected classes count, not unique labels in data
    model = build_model(args.model, num_classes)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
        keras.callbacks.ModelCheckpoint(f"best_{args.model}.h5", save_best_only=True, monitor="val_accuracy", mode="max")
    ]

    history = model.fit([X_train, M_train], Y_train,
                        validation_data=([X_test, M_test], Y_test),
                        epochs=args.epochs, batch_size=args.batch_size,
                        callbacks=callbacks, verbose=1)

    loss, acc = model.evaluate([X_test, M_test], Y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
