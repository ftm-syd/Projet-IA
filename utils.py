import numpy as np
from PIL import Image
import rasterio

def process_sentinel_image(image_path, band=8):
    """Charge une image Sentinel-2 et extrait une bande spécifique"""
    if image_path.lower().endswith(('.tif', '.tiff')):
        with rasterio.open(image_path) as src:
            # Sentinel-2 bandes: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12
            # B8 est le NIR (bande 8)
            band_index = band  # Sentinel-2 bands are 1-based in the file
            if band_index > src.count:
                raise ValueError(f"La bande {band} n'existe pas dans l'image")
            band_data = src.read(band_index)
            return band_data
    else:
        # Pour les formats non-GeoTIFF, charger comme image normale
        img = Image.open(image_path)
        return np.array(img.convert('L'))  # Convertir en niveaux de gris

def create_ndwi(image_path):
    """Calcule l'index NDWI (Normalized Difference Water Index)"""
    with rasterio.open(image_path) as src:
        # B3 (Green) et B8 (NIR)
        green = src.read(3).astype(np.float32)
        nir = src.read(8).astype(np.float32)
        
        # Éviter la division par zéro
        mask = (nir + green) == 0
        ndwi = np.zeros_like(green)
        ndwi[~mask] = (green[~mask] - nir[~mask]) / (green[~mask] + nir[~mask])
        
        # Normaliser entre 0 et 1
        ndwi = (ndwi + 1) / 2
        return ndwi

def normalize_image(image_array):
    """Normalise l'image entre 0 et 1"""
    if image_array.dtype == np.uint8:
        return image_array / 255.0
    elif image_array.dtype == np.uint16:
        return image_array / 65535.0
    else:
        # Pour les images déjà en float
        return (image_array - image_array.min()) / (image_array.max() - image_array.min())