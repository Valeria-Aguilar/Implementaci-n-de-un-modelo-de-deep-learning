# models/rnn_model.py
from tensorflow import keras
from config import MAX_SEQ_LENGTH, NUM_FEATURES

class RNNModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def build_model(self):
        """Modelo optimizado basado en el análisis de resultados"""
        frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
        mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")
        
        # Arquitectura más balanceada
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(64, return_sequences=True, dropout=0.2)
        )(frame_features_input, mask=mask_input)
        
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LSTM(32, dropout=0.1)(x)
        x = keras.layers.BatchNormalization()(x)
        
        # Menos regularización para datos simples
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model([frame_features_input, mask_input], outputs)
        
        # Optimizador con learning rate más bajo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model