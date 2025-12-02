from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
# Temporairement, utilisons l'ancien détecteur en attendant la finalisation
from detector.adaptive_detector import AdaptiveAIDetector
ENHANCED_MODE = False
print("⚠️ Mode Legacy AI Detector (ancien système) - Enhanced en développement")
from detector.utils import setup_project_structure
from detector.workers import extract_features_worker
from pathlib import Path
import os
import tempfile
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import logging
from werkzeug.utils import secure_filename
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, static_folder='../../neon-scan/dist', static_url_path='')
CORS(app, origins=['http://20.123.91.83', 'https://20.123.91.83', 'http://localhost:8000'])  # Enable CORS for specific domains

# Configuration
MODEL_PATH = "models/ai_detector_model.pkl"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global detector instance
detector = None

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/api/setup', methods=['POST'])
def setup():
    setup_project_structure()
    return jsonify({'status': 'structure created'}), 200

@app.route('/api/train', methods=['POST'])
def train():
    real_dir = request.json.get("real_images_folder")
    fake_dir = request.json.get("ai_generated_folder")
    if not real_dir or not fake_dir:
        return jsonify({'error': 'Missing folder'}), 400
    real_paths = list(Path(real_dir).glob("*.jpg")) + list(Path(real_dir).glob("*.png"))
    fake_paths = list(Path(fake_dir).glob("*.jpg")) + list(Path(fake_dir).glob("*.png"))
    X, y = [], []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(extract_features_worker, str(p)): 0 for p in real_paths}
        futures.update({executor.submit(extract_features_worker, str(p)): 1 for p in fake_paths})
        for future in as_completed(futures):
            label = futures[future]
            features = future.result()
            if features is not None:
                X.append(features)
                y.append(label)
    detector = AdaptiveAIDetector()
    report, auc_score = detector.train(X, y)
    detector.save_model(MODEL_PATH)
    return jsonify({'report': report, 'auc': auc_score}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    img_path = request.json.get('image_path')
    if not img_path or not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'Invalid request or model not found'}), 400
    global detector
    if detector is None:
        detector = AdaptiveAIDetector(MODEL_PATH)
    score, label = detector.predict(img_path)
    return jsonify({'score': float(score), 'prediction': label})


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload():
    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            return jsonify({'error': 'Model not found. Please train the model first.'}), 400
        
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded. Please select an image.'}), 400
        
        img_file = request.files['image']
        
        # Check if file is valid
        if img_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(img_file.filename):
            return jsonify({'error': 'Invalid file type. Please upload JPG, PNG, or WebP images.'}), 400
        
        # Create secure filename and save temporarily
        filename = secure_filename(img_file.filename)
        temp_dir = tempfile.mkdtemp()
        img_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{filename}")
        
        # Save the file
        img_file.save(img_path)
        
        # Validate the image file
        import cv2
        import numpy as np
        
        # Try to read the image to validate it
        img = cv2.imread(img_path)
        if img is None:
            # Try alternative methods for corrupted files
            try:
                from PIL import Image
                pil_img = Image.open(img_path)
                pil_img = pil_img.convert('RGB')
                # Save as a clean copy
                clean_path = os.path.join(temp_dir, f"clean_{uuid.uuid4()}.jpg")
                pil_img.save(clean_path, 'JPEG', quality=95)
                img_path = clean_path
                
                # Verify the cleaned image can be read by OpenCV
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError("Unable to read image after cleaning")
                    
            except Exception as img_error:
                shutil.rmtree(temp_dir)
                return jsonify({'error': f'Invalid or corrupted image file: {str(img_error)}'}), 400
        
        # Validate image dimensions and format
        if img.shape[0] < 32 or img.shape[1] < 32:
            shutil.rmtree(temp_dir)
            return jsonify({'error': 'Image too small. Minimum size: 32x32 pixels'}), 400
            
        if img.shape[0] > 4096 or img.shape[1] > 4096:
            shutil.rmtree(temp_dir)
            return jsonify({'error': 'Image too large. Maximum size: 4096x4096 pixels'}), 400
        
        # Load detector if not already loaded
        global detector
        if detector is None:
            detector = AdaptiveAIDetector(MODEL_PATH)
        
        # Make prediction
        score, label = detector.predict(img_path)
        
        # Clean up temp file
        shutil.rmtree(temp_dir)
        
        return jsonify({
            'score': float(score),
            'prediction': label,
            'status': 'success'
        })
        
    except Exception as e:
        logging.error(f"Error in upload: {str(e)}")
        # Clean up temp directory if it exists
        if 'temp_dir' in locals():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None,
        'model_exists': os.path.exists(MODEL_PATH)
    })

# Serve React app
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

# Catch all route for React Router
@app.route('/<path:path>')
def serve_frontend_routes(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    print("Starting AI Image Detector Server...")
    print(f"Backend API: http://localhost:8000/api")
    print(f"Frontend: http://localhost:8000")
    
    # Initialize the model if it exists
    if os.path.exists(MODEL_PATH):
        print("Loading pre-trained model...")
        detector = AdaptiveAIDetector(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print("No pre-trained model found. You'll need to train the model first.")
        
    app.run(host='0.0.0.0', port=8000, debug=True)
