from flask import Flask, request, jsonify
from detector.adaptive_detector import AdaptiveAIDetector
from detector.utils import setup_project_structure
from detector.workers import extract_features_worker
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
MODEL_PATH = "models/ai_detector_model.pkl"
detector = None

@app.route('/setup', methods=['POST'])
def setup():
    setup_project_structure()
    return jsonify({'status': 'structure created'}), 200

@app.route('/train', methods=['POST'])
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

@app.route('/predict', methods=['POST'])
def predict():
    img_path = request.json.get('image_path')
    if not img_path or not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'Invalid request or model not found'}), 400
    global detector
    if detector is None:
        detector = AdaptiveAIDetector(MODEL_PATH)
    score, label = detector.predict(img_path)
    return jsonify({'score': float(score), 'prediction': label})

@app.route('/upload', methods=['POST'])
def upload():
    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'Model not found, please train first'}), 400
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    img_file = request.files['file']
    temp_dir = tempfile.mktemp()
    img_path = os.path.join(temp_dir, str(uuid.uuid4()) + "_" + img_file.filename)
    img_file.save(img_path)
    try:
        global detector
        if detector is None:
            detector = AdaptiveAIDetector(MODEL_PATH)
        score, label = detector.predict(img_path)
        return jsonify({'score': float(score), 'prediction': label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        shutil.rmtree(temp_dir)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
