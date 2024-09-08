from flask import render_template, request, redirect, url_for
from app import app
from app.model import load_model, predict_traffic_sign
import os
from werkzeug.utils import secure_filename
import glob

# Load the model once when the server starts
model = load_model()

UPLOAD_FOLDER = 'app/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def cleanup_uploads():
    """Remove all files in the uploads directory."""
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
    for f in files:
        os.remove(f)

@app.route('/')
def index():
    # Clean up the uploads directory when user returns to the home page
    cleanup_uploads()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        img = request.files['file']
        filename = secure_filename(img.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(filepath)
        
        # Make prediction
        label = predict_traffic_sign(model, filepath)
        
        # Render result page with the image and prediction
        return render_template('result.html', label=label, image_url=url_for('static', filename=f'uploads/{filename}'))
    return redirect(url_for('index'))
