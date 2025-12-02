# models/baseline_model.py
from tensorflow import keras
from config import MAX_SEQ_LENGTH, NUM_FEATURES

class BaselineModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def build_model(self):
        """Construye modelo baseline LSTM simple"""
        frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
        mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")
        
        x = keras.layers.LSTM(64, return_sequences=True)(
            frame_features_input, mask=mask_input
        )
        x = keras.layers.LSTM(32)(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model([frame_features_input, mask_input], outputs)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model