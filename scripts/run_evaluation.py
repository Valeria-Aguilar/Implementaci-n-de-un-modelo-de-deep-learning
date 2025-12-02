#!/usr/bin/env python3
"""
CNN-RNN Action Recognition: Model Evaluation and Predictions

This script evaluates all trained models and generates comprehensive results.
Equivalent to the Jupyter notebook but runs as a command-line script.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

# Custom imports
from config import SELECTED_CLASSES

# Set up plotting (non-interactive)
plt.switch_backend('Agg')  # Use non-interactive backend
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def load_model_safely(model_path):
    """Load a model with error handling"""
    try:
        model = keras.models.load_model(model_path, compile=False)
        print(f"Loaded {model_path}")
        return model
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

def load_and_filter_test_data():
    """Load test data and filter for selected classes"""
    data_dir = "data/processed"

    # Load raw data
    X_test = np.load(os.path.join(data_dir, "test_features.npy"))
    M_test = np.load(os.path.join(data_dir, "test_masks.npy"))
    Y_test = np.load(os.path.join(data_dir, "test_labels.npy"))

    print(f"Raw test data shape: {X_test.shape}")
    print(f"Unique labels in raw data: {len(np.unique(Y_test))} (0-{np.unique(Y_test).max()})")

    # Filter for selected classes (0-4)
    selected_classes = list(range(len(SELECTED_CLASSES)))
    mask = np.isin(Y_test, selected_classes)

    X_test_filtered = X_test[mask]
    M_test_filtered = M_test[mask]
    Y_test_filtered = Y_test[mask]

    print(f"\nFiltered test data shape: {X_test_filtered.shape}")
    print(f"Selected classes: {SELECTED_CLASSES}")

    # Show class distribution
    unique, counts = np.unique(Y_test_filtered, return_counts=True)
    for i, (class_idx, count) in enumerate(zip(unique, counts)):
        print(f"  {SELECTED_CLASSES[class_idx]}: {count} samples")

    return X_test_filtered, M_test_filtered, Y_test_filtered

def evaluate_model(model, name, X_test, M_test, Y_test):
    """Evaluate a single model and return detailed metrics"""
    print(f"\nEvaluating {name} model...")

    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Get predictions
    predictions = model.predict([X_test, M_test], verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = Y_test

    # Calculate metrics
    loss, accuracy = model.evaluate([X_test, M_test], Y_test, verbose=0)

    print(".4f")
    print(".4f")

    # Per-class accuracy
    per_class_acc = {}
    for i, class_name in enumerate(SELECTED_CLASSES):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred[class_mask] == i)
            per_class_acc[class_name] = class_acc
            print(".4f")

    return {
        'name': name,
        'loss': loss,
        'accuracy': accuracy,
        'per_class': per_class_acc,
        'predictions': y_pred,
        'probabilities': predictions
    }

def create_comparison_plots(results, Y_test, output_dir="evaluation_results"):
    """Create comparison plots and save them"""
    os.makedirs(output_dir, exist_ok=True)

    # Create comparison dataframe
    comparison_data = []
    for name, result in results.items():
        row = {
            'Model': name.title(),
            'Overall Accuracy': result['accuracy'],
            'Test Loss': result['loss']
        }
        # Add per-class accuracies
        for class_name, acc in result['per_class'].items():
            row[class_name] = acc
        comparison_data.append(row)

    df_comparison = pd.DataFrame(comparison_data)

    # Plot 1: Overall accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(df_comparison['Model'], df_comparison['Overall Accuracy'])
    plt.title('Overall Test Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    for i, v in enumerate(df_comparison['Overall Accuracy']):
        plt.text(i, v + 0.01, '.1%', ha='center', va='bottom')
    plt.savefig(os.path.join(output_dir, 'overall_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Per-class accuracy heatmap
    per_class_cols = [col for col in df_comparison.columns if col not in ['Model', 'Overall Accuracy', 'Test Loss']]
    per_class_data = df_comparison[per_class_cols].T
    per_class_data.columns = df_comparison['Model']

    plt.figure(figsize=(10, 8))
    sns.heatmap(per_class_data, annot=True, fmt='.3f', cmap='YlOrRd', cbar=True)
    plt.title('Per-Class Accuracy Heatmap')
    plt.ylabel('Action Class')
    plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Confusion matrices
    fig, axes = plt.subplots(1, len(results), figsize=(18, 6))
    if len(results) == 1:
        axes = [axes]

    for i, (name, result) in enumerate(results.items()):
        # Create confusion matrix
        cm = confusion_matrix(Y_test, result['predictions'])

        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=SELECTED_CLASSES, yticklabels=SELECTED_CLASSES, ax=axes[i])
        axes[i].set_title('.1%')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return df_comparison

def predict_sample(model, name, X_sample, M_sample, Y_sample_true):
    """Make predictions on sample data and show results"""
    print(f"\n{name.title()} Model Predictions:")

    # Get predictions
    predictions = model.predict([X_sample, M_sample], verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    results = []
    for i, (true_label, pred_label, probs) in enumerate(zip(Y_sample_true, y_pred, predictions)):
        true_class = SELECTED_CLASSES[true_label]
        pred_class = SELECTED_CLASSES[pred_label]
        confidence = probs[pred_label]
        correct = "Correct" if true_label == pred_label else "Wrong"

        print(f"  Sample {i+1}: {true_class} → {pred_class} ({confidence:.1%}) {correct}")

        results.append({
            'sample': i+1,
            'true': true_class,
            'predicted': pred_class,
            'confidence': confidence,
            'correct': true_label == pred_label
        })

    return results

def main():
    print("CNN-RNN Action Recognition: Model Evaluation and Predictions")
    print("=" * 70)

    # Load models
    print("\nLoading trained models...")
    model_paths = {
        'baseline': 'best_baseline.h5',
        'optimized': 'best_optimized.h5',
        'improved': 'best_improved.h5'
    }

    models = {}
    for name, path in model_paths.items():
        model = load_model_safely(path)
        if model is not None:
            models[name] = model

    print(f"Successfully loaded {len(models)}/{len(model_paths)} models")

    if not models:
        print("No models loaded. Exiting.")
        return

    # Load test data
    print("\nLoading and preprocessing test data...")
    X_test, M_test, Y_test = load_and_filter_test_data()

    # Evaluate all models
    print("\nEvaluating models on test data...")
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, name, X_test, M_test, Y_test)

    # Create comparison plots
    print("\nGenerating comparison plots...")
    df_comparison = create_comparison_plots(results, Y_test)
    print("   Plots saved to 'evaluation_results/' directory")

    # Sample predictions
    print("\nTesting predictions on sample data...")
    np.random.seed(42)  # For reproducible results
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    X_sample = X_test[sample_indices]
    M_sample = M_test[sample_indices]
    Y_sample_true = Y_test[sample_indices]

    all_predictions = {}
    for name, model in models.items():
        all_predictions[name] = predict_sample(model, name, X_sample, M_sample, Y_sample_true)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    print("\nModel Performance Ranking:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for i, (name, result) in enumerate(sorted_results, 1):
        print(f"  {i}. {name.title()}: {result['accuracy']:.1%} accuracy")

    print("\nDetailed Results:")
    print(df_comparison.to_string(index=False, float_format='.4f'))

    print("\nResults saved:")
    print("  • evaluation_results/overall_accuracy.png")
    print("  • evaluation_results/per_class_accuracy.png")
    print("  • evaluation_results/confusion_matrices.png")

    # Best model recommendation
    best_model = sorted_results[0]
    print(f"\nRECOMMENDATION: Use {best_model[0].title()} model for best overall performance")

    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
