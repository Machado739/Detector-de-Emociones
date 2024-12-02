from __future__ import print_function
from PIL import Image, ImageTk
import tkinter as tki
import threading
import cv2
import os
import numpy as np
from machineLearning.predictor import Predictor


class PhotoApp:
    def __init__(self, vs, outputPath):
        self.vs = vs
        self.outputPath = outputPath
        self.frame = None
        self.thread = None
        self.stopEvent = None

        # Validar o crear el directorio de salida
        self.validateOutputPath()

        # Inicializar Predictor
        try:
            self.predictor = Predictor(
                model_path="/home/jaime/Desktop/Detector de Emociones/machineLearning/modelo_mejorado_emociones.h5"
            )
            print("[INFO] Predictor cargado correctamente.")
        except Exception as e:
            print(f"[ERROR] Error al cargar el Predictor: {e}")
            self.predictor = None

        # Configurar la ventana principal
        self.root = tki.Tk()
        self.root.geometry("800x600")
        self.root.title("Detección de Emociones")

        # Panel de la cámara
        self.panel = None

        # Cuadro para mostrar emociones
        self.emotion_label = tki.Label(
            self.root, text="Emoción detectada: Ninguna", bg="white", fg="black", font=("Arial", 14)
        )
        self.emotion_label.place(relx=0.0, rely=0.9, relheight=0.1, relwidth=1.0)

        # Botones para iniciar/detener la cámara
        self.btn_start = tki.Button(self.root, text="Iniciar Cámara", bg="green", command=self.startCamera)
        self.btn_start.place(relx=0.3, rely=0.8, relheight=0.1, relwidth=0.4)

        self.btn_close = tki.Button(self.root, text="Cerrar", bg="red", command=self.onClose)
        self.btn_close.place(relx=0.8, rely=0.9, relheight=0.1, relwidth=0.2)

        self.stopEvent = threading.Event()

    def validateOutputPath(self):
        """Valida o crea el directorio de salida."""
        if not os.path.exists(self.outputPath):
            os.makedirs(self.outputPath)
            print(f"[INFO] Directorio de salida creado: {self.outputPath}")

    def videoLoop(self):
        """Bucle principal para capturar y procesar video en tiempo real."""
        try:
            while not self.stopEvent.is_set():
                self.frame = self.vs.read()

                # Procesar solo si la cámara está activa
                if not self.stopEvent.is_set():
                    emotion = self.detectEmotion(self.frame)
                    self.emotion_label.config(text=f"Emoción detectada: {emotion if emotion else 'Ninguna'}")

                    # Convertir el fotograma para tkinter
                    image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    image = ImageTk.PhotoImage(image)

                    if self.panel is None:
                        self.panel = tki.Label(image=image)
                        self.panel.image = image
                        self.panel.place(relx=0.0, rely=0.0, relheight=0.8, relwidth=1.0)
                    else:
                        self.panel.configure(image=image)
                        self.panel.image = image

        except RuntimeError as e:
            print(f"[ERROR] Error en videoLoop: {e}")

    def detectEmotion(self, frame):
        """Detecta la emoción en un fotograma."""
        try:
            if self.predictor is None or self.predictor.model is None:
                return "Error: Modelo no cargado"

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            resized = resized.astype("float32") / 255.0
            resized = np.expand_dims(resized, axis=(0, -1))

            probabilities = self.predictor.model.predict(resized)
            emotion_index = np.argmax(probabilities)
            mapper = {0: "Felicidad", 1: "Tristeza", 2: "Neutral"}
            return mapper.get(emotion_index, "Desconocido")
        except Exception as e:
            print(f"[ERROR] Error en detectEmotion: {e}")
            return None

    def startCamera(self):
        """Inicia la cámara y el bucle de video."""
        try:
            if not self.thread or not self.thread.is_alive():
                self.stopEvent.clear()
                self.thread = threading.Thread(target=self.videoLoop, args=(), daemon=True)
                self.thread.start()
                self.btn_start.config(text="Detener Cámara", command=self.stopCamera, bg="orange")
        except Exception as e:
            print(f"[ERROR] No se pudo iniciar la cámara: {e}")

    def stopCamera(self):
        """Pausa la cámara y detiene el procesamiento."""
        self.stopEvent.set()

        # Crear un fondo negro para tapar el último fotograma
        black_frame = np.zeros((480, 640, 3), dtype="uint8")
        image = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        if self.panel:
            self.panel.configure(image=image)
            self.panel.image = image
        else:
            self.panel = tki.Label(image=image)
            self.panel.image = image
            self.panel.place(relx=0.0, rely=0.0, relheight=0.8, relwidth=1.0)

        # Cambiar el estado del botón
        self.btn_start.config(text="Iniciar Cámara", command=self.startCamera, bg="green")

        print("[INFO] Cámara detenida y panel actualizado.")

    def onClose(self):
        """Cierra la aplicación y libera recursos."""
        print("[INFO] Cerrando...")
        # Detener la cámara antes de cerrar
        if not self.stopEvent.is_set():
            self.stopCamera()

        # Liberar recursos de la cámara
        if hasattr(self.vs, "stop"):
            self.vs.stop()
        elif hasattr(self.vs, "release"):
            self.vs.release()

        # Cerrar la ventana de Tkinter
        self.root.destroy()
        print("[INFO] Aplicación cerrada correctamente.")
