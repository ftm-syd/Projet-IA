import numpy as np
import cv2
import rasterio
import os
from tensorflow.keras.models import load_model

# Charger le modèle
model = load_model("model/unet_membrane.keras")

def preprocess_band(band_path):
    with rasterio.open(band_path) as src:
        band = src.read(1)

    # Normalisation
    band = band / 255.0

    # Redimensionner à 256x256
    band = cv2.resize(band, (256, 256))

    # Ajouter les dimensions : (H, W, 1) puis batch
    band = np.expand_dims(band, axis=-1)  # (256, 256, 1)
    band = np.expand_dims(band, axis=0)   # (1, 256, 256, 1)

    return band

def run_model(band_path):
    img = preprocess_band(band_path)
    prediction = model.predict(img)[0, :, :, 0]  # (H, W)

    print("Valeurs min/max de la prédiction :", prediction.min(), prediction.max())

    mask = (prediction > 0.7).astype(np.uint8) * 255



    result_path = os.path.join("static/results", "predicted_mask.png")
    cv2.imwrite(result_path, mask)

    return result_path



