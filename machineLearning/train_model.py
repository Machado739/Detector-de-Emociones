import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Configuración de rutas
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")
OUTPUT_MODEL = "modelo_mejorado_emociones.h5"

# Verifica que las carpetas existan
for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"[ERROR] La carpeta {directory} no existe.")

# 1. Preprocesamiento de datos
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Generadores de datos
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(48, 48),
    batch_size=32,
    class_mode="categorical",
    color_mode="grayscale",
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(48, 48),
    batch_size=32,
    class_mode="categorical",
    color_mode="grayscale",
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(48, 48),
    batch_size=32,
    class_mode="categorical",
    color_mode="grayscale",
)
# Verifica el shape de los datos
batch = next(train_generator)
print(f"[INFO] Forma de las imágenes: {batch[0].shape}")
print(f"[INFO] Forma de las etiquetas: {batch[1].shape}")
# Balanceo de clases (opcional)
class_weights = {i: 1.0 for i in range(train_generator.num_classes)}

# 2. Diseño del modelo
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation="softmax"),
])

# 3. Compilación del modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# Resumen del modelo
model.summary()

# 4. Entrenamiento del modelo con Early Stopping
EPOCHS = 100

early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
)

# 5. Evaluación del modelo
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Precisión en el conjunto de prueba: {test_accuracy*100:.2f}%")

# 6. Guardar el modelo
model.save(OUTPUT_MODEL)
print(f"[INFO] Modelo guardado en: {OUTPUT_MODEL}")

# Guardar el historial de entrenamiento en un archivo JSON
with open("historial_entrenamiento.json", "w") as f:
    json.dump(history.history, f)
print("[INFO] Historial de entrenamiento guardado en 'historial_entrenamiento.json'")

# Guardar resultados de prueba en un archivo
with open("resultados_prueba.txt", "w") as f:
    f.write(f"Pérdida en prueba: {test_loss:.4f}\n")
    f.write(f"Precisión en prueba: {test_accuracy*100:.2f}%\n")
print("[INFO] Resultados de prueba guardados en 'resultados_prueba.txt'")

# 7. Graficar los resultados
plt.figure(figsize=(12, 6))

# Gráfica de precisión
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Precisión en entrenamiento")
plt.plot(history.history["val_accuracy"], label="Precisión en validación")
plt.title("Precisión")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.legend()

# Gráfica de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Pérdida en entrenamiento")
plt.plot(history.history["val_loss"], label="Pérdida en validación")
plt.title("Pérdida")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend()

plt.tight_layout()
plt.show()
