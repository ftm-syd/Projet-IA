import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np
from PIL import Image
import tensorflow as tf
from utils import process_sentinel_image, create_ndwi, normalize_image

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'tif', 'tiff', 'png', 'jpg', 'jpeg'}

# Charger le modèle UNet
model = tf.keras.models.load_model('unet_membrane.keras')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Vérifier si le fichier a été envoyé
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Sauvegarder le fichier uploadé
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            # Traiter l'image
            try:
                # Charger et prétraiter l'image Sentinel-2
                if request.form.get('use_ndwi') == 'on':
                    # Si NDWI est sélectionné
                    img_array = create_ndwi(upload_path)
                else:
                    # Sinon utiliser la bande B8 directement
                    img_array = process_sentinel_image(upload_path, band=8)
                
                # Normaliser l'image
                img_array = normalize_image(img_array)
                
                # Ajouter une dimension de batch et de canal
                img_array = np.expand_dims(img_array, axis=0)
                img_array = np.expand_dims(img_array, axis=-1)
                
                # Faire la prédiction
                prediction = model.predict(img_array)
                prediction_mask = (prediction.squeeze() > 0.5).astype(np.uint8) * 255
                
                # Sauvegarder le résultat
                result_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                Image.fromarray(prediction_mask).save(result_path)
                
                return render_template('index.html', 
                                     original_image=upload_path, 
                                     result_image=result_path)
            
            except Exception as e:
                return render_template('index.html', error=str(e))
    
    return render_template('index.html')

if __name__ == '__main__':
    # Créer les dossiers s'ils n'existent pas
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    app.run(debug=True)
