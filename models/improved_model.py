# models/improved_model.py
import tensorflow as tf
from tensorflow import keras
from config import MAX_SEQ_LENGTH, NUM_FEATURES

class ImprovedModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build_model(self):
        """Modelo mejorado con técnicas de regularización avanzadas y arquitectura optimizada"""
        frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
        mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

        # Regularización L1/L2
        l2_reg = keras.regularizers.l2(1e-4)
        l1_l2_reg = keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)

        # Arquitectura mejorada con atención y conexiones residuales
        # Primera capa LSTM bidireccional con regularización aumentada
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.1,
                             kernel_regularizer=l2_reg)
        )(frame_features_input, mask=mask_input)

        # Batch normalization y dropout
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.4)(x)

        # Mecanismo de atención simple
        attention_weights = keras.layers.Dense(1, activation='tanh')(x)
        attention_weights = keras.layers.Flatten()(attention_weights)
        attention_weights = keras.layers.Activation('softmax')(attention_weights)
        attention_weights = keras.layers.RepeatVector(256)(attention_weights)  # 128*2 para bidireccional
        attention_weights = keras.layers.Permute([2, 1])(attention_weights)

        x_attended = keras.layers.Multiply()([x, attention_weights])
        x_attended = keras.layers.GlobalAveragePooling1D()(x_attended)

        # Conexión residual al combinar atención con LSTM directo
        x_direct = keras.layers.LSTM(64, dropout=0.2, kernel_regularizer=l1_l2_reg)(x)
        x_combined = keras.layers.Concatenate()([x_attended, x_direct])
        x_combined = keras.layers.BatchNormalization()(x_combined)

        # Capas densas con regularización progresiva
        x = keras.layers.Dense(256, activation='relu', kernel_regularizer=l1_l2_reg)(x_combined)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.5)(x)

        x = keras.layers.Dense(128, activation='relu', kernel_regularizer=l2_reg)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.4)(x)

        # Capa intermedia con activación ELU para mejor gradiente
        x = keras.layers.Dense(64, activation='elu', kernel_regularizer=l2_reg)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)

        # Capa de salida con regularización suave
        outputs = keras.layers.Dense(self.num_classes, activation='softmax',
                                   kernel_regularizer=keras.regularizers.l2(1e-5))(x)

        model = keras.Model([frame_features_input, mask_input], outputs)

        # Optimizador con gradient clipping y learning rate inicial
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0  # Gradient clipping
        )

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
