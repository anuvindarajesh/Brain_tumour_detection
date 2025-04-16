import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image

# Initialize Flask app
app = Flask(__name__, template_folder='.')  # Look for templates in the same directory

# Load trained U-Net model
MODEL_PATH = "unet_brain.h5"
model = load_model(MODEL_PATH, compile=False)

# Configure upload and result folders
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((256, 256))  # Convert to RGB
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Reshape to (1, 256, 256, 3)
    return img_array

# Function to post-process and save the segmented result
def postprocess_and_save(result, filename):
    result_img = (result.squeeze() * 255).astype(np.uint8)  # Convert to image format
    result_pil = Image.fromarray(result_img)
    result_path = os.path.join(RESULT_FOLDER, filename)
    result_pil.save(result_path)
    return result_path

# Function for quantitative analysis
def analyze_tumor(input_image_path, result_image_path):
    # Load MRI (grayscale) and segmentation mask
    mri_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    segmentation_mask = cv2.imread(result_image_path, cv2.IMREAD_GRAYSCALE)

    # Check if images loaded correctly
    if mri_image is None or segmentation_mask is None:
        return "Error loading images", "N/A"

    # Threshold the segmentation mask to extract tumor (assuming white areas represent tumors)
    _, tumor_mask = cv2.threshold(segmentation_mask, 127, 255, cv2.THRESH_BINARY)

    # Compute volumes (count non-zero pixels)
    tumor_volume = np.count_nonzero(tumor_mask)  # Tumor area in pixels
    brain_volume = np.count_nonzero(mri_image)  # Brain area in pixels

    # Compute Tumor-to-Brain Ratio (TBR)
    tbr = (tumor_volume / brain_volume) * 10 if brain_volume > 0 else 0

    # Classify severity
    if tbr == 0.0:
        severity = "No Tumor"
    elif tbr < 0.1:
        severity = "Low Tumor Burden"
    elif 0.1 <= tbr < 0.3:
        severity = "Moderate Tumor Burden"
    else:
        severity = "High Tumor Burden"

    return round(tbr, 4), severity

@app.route('/')
def home():
    return render_template('index.html')  # Loads index.html from the same directory

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Process image
    img_array = preprocess_image(filepath)
    result = model.predict(img_array)
    result_path = postprocess_and_save(result, filename)

    # Perform quantitative analysis
    tumor_brain_ratio, severity_level = analyze_tumor(filepath, result_path)
    
    return render_template('result.html', 
                           input_image=filepath, 
                           result_image=result_path, 
                           tumor_brain_ratio=tumor_brain_ratio, 
                           severity_level=severity_level)

if __name__ == '__main__':
    app.run(debug=True)
