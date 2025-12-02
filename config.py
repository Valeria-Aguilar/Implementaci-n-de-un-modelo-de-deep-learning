# config.py
SELECTED_CLASSES = [
    'Basketball', 
    'BasketballDunk', 
    'ApplyEyeMakeup', 
    'ApplyLipstick', 
    'Archery'
]

# Hyperparameters
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048 + (17 * 3)  # CNN (2048) + skeleton (17 keypoints * 3 features each)
IMG_SIZE = 224