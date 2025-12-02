# CNN-RNN Action Recognition Model

This project implements a deep learning model for action recognition using a combination of CNN (InceptionV3) and RNN (LSTM/GRU) architectures, trained on the UCF101 dataset with skeleton pose estimation features.

## Features

- **CNN Feature Extraction**: Uses pre-trained InceptionV3 for spatial feature extraction from video frames (2048 features per frame)
- **Skeleton Analysis**: Incorporates 2D pose estimation keypoints (17 keypoints × 3 features = 51 features per frame)
- **RNN Modeling**: LSTM/GRU networks for temporal sequence modeling (20-frame sequences)
- **Multi-Model Support**: Three model architectures (Baseline, Optimized, Improved)
- **Video Prediction**: Predict actions from single video files through command line
- **Comprehensive Evaluation**: Per-class accuracy analysis and comparative model evaluation

## Supported Actions

- Basketball
- BasketballDunk
- ApplyEyeMakeup
- ApplyLipstick
- Archery

## Current Model Performance

| Model      | Overall Accuracy | Basketball | BasketballDunk | ApplyEyeMakeup | ApplyLipstick | Archery |
|------------|------------------|------------|----------------|----------------|---------------|---------|
| **Baseline** | **65.57%**    | **75.00%** | 68.75%        | 60.98%        | **94.29%**   | 22.58% |
| **Optimized**| 58.47%        | 34.09%    | **100.00%**   | 53.66%        | 85.71%       | 25.81% |
| **Improved**| 43.72%        | 63.64%    | 31.25%        | 26.83%        | 0.00%        | **100.00%** |

*Latest evaluation results from `scripts/run_evaluation.py`*

### Key Insights:
- **Baseline model** performs best overall (65.57%) with strong performance on ApplyLipstick (94.29%)
- **Optimized model** excels at BasketballDunk recognition (100%) but struggles with Basketball (34.09%)
- **Improved model** shows specialization with perfect Archery recognition (100%) but fails completely on ApplyLipstick (0%)
- Each model has different strengths, suggesting potential for ensemble approaches

### Generated Files:
Running `python scripts/run_evaluation.py` creates:
- `evaluation_results/overall_accuracy.png` - Accuracy comparison bar chart
- `evaluation_results/per_class_accuracy.png` - Per-class accuracy heatmap
- `evaluation_results/confusion_matrices.png` - Confusion matrices for all models

*Note: Models are trained and evaluated on 5 selected action classes from the UCF101 dataset.*

**Model Details:**
- **Baseline**: Simple LSTM network with basic dropout regularization
- **Optimized**: Bidirectional LSTM with batch normalization and progressive dropout
- **Improved**: Advanced architecture with attention mechanism, L1/L2 regularization, residual connections, and gradient clipping

## Quick Start

### 1) Setup Environment (optional)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Generate Features
Extract features from videos and skeleton annotations:
```bash
python scripts/generate_features.py
```
This will process the UCF101 dataset and create training features using InceptionV3.

### 3) Train Models

Train the optimized model (recommended):
```bash
python scripts/train_model.py --model optimized --epochs 30 --batch_size 32
```

Or train the baseline model:
```bash
python scripts/train_model.py --model baseline --epochs 30 --batch_size 32
```

Train the improved model with advanced regularization:
```bash
python scripts/train_model.py --model improved --epochs 30 --batch_size 32
```

*Note: Training automatically filters data to only include the 5 selected action classes.*

### 4) Evaluate Models
Evaluate trained models on test data:
```bash
python scripts/evaluate_model.py best_baseline.h5
python scripts/evaluate_model.py best_optimized.h5
python scripts/evaluate_model.py best_improved.h5
```

### 5) Comprehensive Evaluation
Run full evaluation with comparative analysis and plots:
```bash
python scripts/run_evaluation.py
```
This generates detailed reports and saves visualization plots to `evaluation_results/` directory.

### 5) Predict on Videos
Make predictions on new video files:
```bash
python scripts/predict.py best_baseline.h5 path/to/your/video.mp4
```

### 6) Quick Demo
Test the model with random features:
```bash
python demo.py --model best_baseline.h5
```

## Project Structure

```
├── config.py                 # Configuration and hyperparameters
├── requirements.txt          # Python dependencies
├── demo.py                   # Quick prediction demo with random data
├── setup.py                  # Package installation script
├── .gitignore                # Version control exclusions
├── README.md                 # Comprehensive documentation
├── scripts/
│   ├── generate_features.py  # Feature extraction pipeline
│   ├── train_model.py        # Model training script
│   ├── evaluate_model.py     # Model evaluation script
│   ├── run_evaluation.py     # Comprehensive evaluation script
│   └── predict.py            # Video prediction script
├── src/
│   ├── data_loader.py        # Data loading utilities
│   └── feature_extractor.py  # CNN and skeleton feature extraction
├── models/
│   ├── baseline_model.py     # Simple baseline architecture
│   ├── rnn_model.py          # Optimized RNN model
│   └── improved_model.py     # Advanced model with attention and regularization
├── data/
│   ├── processed/            # Processed features and labels
│   └── *.pkl                 # Raw dataset annotations (excluded from git)
└── __pycache__/              # Python bytecode (excluded from git)
```

## Requirements

- Python 3.8+
- TensorFlow 2.8+
- OpenCV 4.5+
- NumPy 1.21+
- Pandas 1.3+
- Scikit-learn 1.0+
- Matplotlib 3.5+
- Jupyter (for notebooks)

## Data Preparation

1. Download UCF101 dataset videos and place them in `data/videos/`
2. Ensure skeleton annotations are available in `data/ucf101_2d.pkl`
3. Run feature generation script

## Model Architectures

- **Baseline**: Simple LSTM network with basic dropout regularization
- **Optimized**: Bidirectional LSTM with batch normalization and progressive dropout
- **Improved**: Advanced LSTM with attention mechanism, L1/L2 regularization, and residual connections

## Training Tips

- Use GPU for faster feature extraction and training
- Adjust batch size based on available memory
- Monitor validation accuracy for early stopping
- Experiment with different sequence lengths in `config.py`
