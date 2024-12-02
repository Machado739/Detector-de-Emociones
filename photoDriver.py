from photoApp import PhotoApp
from imutils.video import WebcamVideoStream
from machineLearning.predictor import Predictor
import argparse
import time

# Parsear los argumentos
ap = argparse.ArgumentParser()
ap.add_argument(
    "-o",
    "--output",
    required=True,
    help="Path to output directory to store the pictures",
)
args = vars(ap.parse_args())

# Ruta al modelo de emociones
MODEL_PATH = "/home/jaime/Desktop/Detector de Emociones/machineLearning/model.h5"

print("[INFO] Cargando el modelo y preparando la cámara...")

try:
    # Inicializar el predictor
    predictor = Predictor(model_path=MODEL_PATH)

    # Inicializar la cámara
    vs = WebcamVideoStream(src=0).start()
    time.sleep(2.0)

    # Inicializar la aplicación
    print("[INFO] Inicializando la aplicación PhotoApp...")
    pba = PhotoApp(vs, args["output"])
    pba.root.mainloop()
except Exception as e:
    print(f"[ERROR] Ocurrió un error durante la ejecución: {e}")
finally:
    print("[INFO] Cerrando el programa.")
    if 'vs' in locals():
        vs.stop()
