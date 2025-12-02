# main.py
import argparse
import numpy as np
import os
import json
from tensorflow import keras
from config import SELECTED_CLASSES

class ActionPredictor:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo '{model_path}' no encontrado.")
        
        self.model = keras.models.load_model(model_path)
        self.class_names = SELECTED_CLASSES
        
        print("✅ Modelo cargado exitosamente!")

def main():
    parser = argparse.ArgumentParser(description='UCF101 Action Recognition')
    parser.add_argument('--model', default='best_model.h5', help='Ruta al modelo entrenado')
    
    args = parser.parse_args()
    
    try:
        predictor = ActionPredictor(args.model)
        
        # Predicción de ejemplo
        features = np.random.normal(0, 1, (1, 20, 2048))
        mask = np.ones((1, 20), dtype=bool)
        
        predictions = predictor.model.predict([features, mask], verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        print(f"\n🎯 Predicción de ejemplo:")
        print(f"   Acción: {predictor.class_names[predicted_class]}")
        print(f"   Confianza: {confidence:.2%}")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Ejecuta primero: python scripts/train_model.py")

if __name__ == "__main__":
    main()