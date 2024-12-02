import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Cargar el modelo
model = load_model(
    "/home/jaime/Desktop/Detector de Emociones/machineLearning/modelo_mejorado_emociones.h5",
    compile=False  # No necesitamos compilar el modelo para predicción
)

class Predictor:
    def __init__(self, model_path):
        # Intentar cargar el modelo al inicializar la clase
        try:
            self.model = load_model(model_path, compile=False)
            print("[INFO] Modelo cargado correctamente en Predictor.")
        except Exception as e:
            print(f"[ERROR] No se pudo cargar el modelo: {e}")
            self.model = None

    def resize(self, src):
        """Preprocesa la imagen para hacerla compatible con el modelo."""
        try:
            print(f"[INFO] Procesando la imagen: {src}")
            # Leer la imagen en escala de grises
            large_img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            if large_img is None:
                raise FileNotFoundError(f"[ERROR] No se pudo leer la imagen en: {src}")
            
            # Redimensionar la imagen
            resize_img = cv2.resize(large_img, (48, 48))
            image = resize_img.astype("float32") / 255.0
            image = np.expand_dims(image, axis=-1)  # Añadir canal
            image = np.expand_dims(image, axis=0)  # Añadir dimensión batch
            print("[INFO] Imagen procesada correctamente.")
            return image
        except Exception as e:
            print(f"[ERROR] Error durante el procesamiento de la imagen: {e}")
            return None

    def predict(self, src):
        """Realiza la predicción de la emoción basada en la imagen."""
        try:
            if self.model is None:
                print("[ERROR] Modelo no cargado. No se puede hacer la predicción.")
                return None

            # Preprocesar la imagen
            image = self.resize(src)
            if image is None:
                print("[ERROR] No se pudo procesar la imagen. Predicción abortada.")
                return None

            # Realizar la predicción
            probabilities = self.model.predict(image)
            prediction = np.argmax(probabilities, axis=1)[0]

            # Mapeo de emociones
            mapper = {
                0: "Felicidad",
                1: "Tristeza",
                2: "Neutral",
            }
            emotion = mapper.get(prediction, "Desconocido")
            print(f"[INFO] Predicción completada. Emoción detectada: {emotion}")
            return emotion
        except Exception as e:
            print(f"[ERROR] Error durante la predicción: {e}")
            return None

# Crear una instancia del predictor con la ruta al modelo
predictor = Predictor("/home/jaime/Desktop/Detector de Emociones/machineLearning/modelo_mejorado_emociones.h5")
