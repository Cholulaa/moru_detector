from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import shutil
import uuid
import sys
import logging
from werkzeug.utils import secure_filename
from pathlib import Path
import time
import json

# Import du nouveau détecteur ultra-avancé
try:
    from detector.ultra_enhanced_detector import UltraEnhancedAIDetector
    ULTRA_MODE = True
    print("🚀 Mode Ultra-Avancé activé - Détecteur IA v4.0")
except ImportError as e:
    print(f"⚠️ Impossible de charger le détecteur ultra-avancé: {e}")
    from detector.adaptive_detector import AdaptiveAIDetector
    ULTRA_MODE = False
    print("⚠️ Mode Legacy activé - Détecteur IA v3.0")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, static_folder='../../neon-scan/dist', static_url_path='')
CORS(app, origins=['http://20.123.91.83', 'https://20.123.91.83', 'http://localhost:8000'])

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ultra_ai_detector_20251205_133124.pkl") if ULTRA_MODE else os.path.join(BASE_DIR, "models", "ai_detector_model.pkl")
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Global detector instance
detector = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/api_ultra.log')
    ]
)
logger = logging.getLogger('ultra_api')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/setup', methods=['POST'])
def setup():
    """Configuration initiale du système"""
    try:
        from detector.utils import setup_project_structure
        setup_project_structure()
        return jsonify({
            'status': 'success',
            'message': 'Structure projet créée',
            'ultra_mode': ULTRA_MODE
        }), 200
    except Exception as e:
        logger.error(f"Erreur setup: {e}")
        return jsonify({'error': f'Erreur setup: {str(e)}'}), 500


@app.route('/api/train', methods=['POST'])
def train():
    """Entraînement du modèle (ultra-avancé ou legacy)"""
    try:
        data = request.get_json()
        real_dir = data.get("real_images_folder")
        fake_dir = data.get("ai_generated_folder")
        
        if not real_dir or not fake_dir:
            return jsonify({'error': 'Dossiers manquants'}), 400
        
        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            return jsonify({'error': 'Dossiers non trouvés'}), 400
        
        logger.info(f"Début entraînement - Mode: {'Ultra' if ULTRA_MODE else 'Legacy'}")
        start_time = time.time()
        
        if ULTRA_MODE:
            # Entraînement ultra-avancé
            config = {
                'feature_extraction': {
                    'use_wavelets': True,
                    'use_gabor_filters': True,
                    'multiscale_analysis': True,
                    'wavelet_types': ['db4', 'haar'],
                    'gabor_frequencies': [0.1, 0.2, 0.3],
                    'gabor_orientations': [0, 45, 90, 135]
                },
                'model_training': {
                    'ensemble_methods': ['xgboost', 'rf', 'lightgbm'],
                    'use_stacking': True,
                    'use_feature_selection': True,
                    'cross_validation_folds': 3,  # Réduire pour la vitesse
                    'hyperparameter_tuning': False,
                    'class_weight': 'balanced'
                },
                'preprocessing': {
                    'image_sizes': [(256, 256)],  # Une seule taille pour la vitesse
                    'normalization_methods': ['standard', 'robust'],
                    'noise_reduction': True,
                    'contrast_enhancement': True
                },
                'performance': {
                    'parallel_processing': True,
                    'max_workers': min(4, os.cpu_count() or 1),
                    'batch_processing': True,
                    'memory_optimization': True
                }
            }
            
            detector = UltraEnhancedAIDetector(config)
            
            # Collection des images
            from pathlib import Path
            real_paths = list(Path(real_dir).glob("*.jpg")) + list(Path(real_dir).glob("*.png"))
            fake_paths = list(Path(fake_dir).glob("*.jpg")) + list(Path(fake_dir).glob("*.png"))
            
            # Limiter le nombre d'images pour la démo
            max_images = 500
            real_paths = real_paths[:max_images]
            fake_paths = fake_paths[:max_images]
            
            logger.info(f"Images à traiter: {len(real_paths)} réelles, {len(fake_paths)} IA")
            
            # Extraction des features
            X, y = [], []
            total_images = len(real_paths) + len(fake_paths)
            processed = 0
            
            for path in real_paths[:max_images//2]:  # Limiter encore plus pour la vitesse
                try:
                    features = detector.extract_ultra_features(str(path))
                    X.append(features)
                    y.append(0)
                    processed += 1
                    if processed % 50 == 0:
                        logger.info(f"Traité {processed}/{total_images} images")
                except Exception as e:
                    logger.warning(f"Erreur extraction {path}: {e}")
            
            for path in fake_paths[:max_images//2]:
                try:
                    features = detector.extract_ultra_features(str(path))
                    X.append(features)
                    y.append(1)
                    processed += 1
                    if processed % 50 == 0:
                        logger.info(f"Traité {processed}/{total_images} images")
                except Exception as e:
                    logger.warning(f"Erreur extraction {path}: {e}")
            
            if len(X) < 10:
                return jsonify({'error': 'Pas assez d\'images valides pour l\'entraînement'}), 400
            
            # Entraînement
            report, auc_score = detector.train_ultra_model(X, y)
            detector.save_ultra_model(MODEL_PATH)
            
        else:
            # Entraînement legacy
            from detector.workers import extract_features_worker
            from concurrent.futures import ProcessPoolExecutor, as_completed
            
            real_paths = list(Path(real_dir).glob("*.jpg")) + list(Path(real_dir).glob("*.png"))
            fake_paths = list(Path(fake_dir).glob("*.jpg")) + list(Path(fake_dir).glob("*.png"))
            
            X, y = [], []
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(extract_features_worker, str(p)): 0 for p in real_paths[:100]}
                futures.update({executor.submit(extract_features_worker, str(p)): 1 for p in fake_paths[:100]})
                
                for future in as_completed(futures):
                    label = futures[future]
                    features = future.result()
                    if features is not None:
                        X.append(features)
                        y.append(label)
            
            detector = AdaptiveAIDetector()
            report, auc_score = detector.train(X, y)
            detector.save_model(MODEL_PATH)
        
        training_time = time.time() - start_time
        
        result = {
            'status': 'success',
            'report': report,
            'auc': float(auc_score),
            'training_time': training_time,
            'samples_processed': len(X),
            'mode': 'Ultra-Avancé' if ULTRA_MODE else 'Legacy'
        }
        
        logger.info(f"Entraînement terminé: AUC={auc_score:.4f}, temps={training_time:.1f}s")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Erreur entraînement: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Erreur entraînement: {str(e)}'}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Prédiction sur une image"""
    try:
        img_path = request.json.get('image_path')
        if not img_path or not os.path.exists(img_path):
            return jsonify({'error': 'Chemin image invalide'}), 400
        
        if not os.path.exists(MODEL_PATH):
            return jsonify({'error': 'Modèle non trouvé. Entraînez d\'abord le modèle.'}), 400
        
        global detector
        if detector is None:
            if ULTRA_MODE:
                detector = UltraEnhancedAIDetector()
                detector.load_ultra_model(MODEL_PATH)
            else:
                detector = AdaptiveAIDetector(MODEL_PATH)
        
        if ULTRA_MODE:
            result = detector.predict_ultra(img_path)
            return jsonify({
                'score': result.score,
                'prediction': result.prediction,
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'image_quality': result.image_quality,
                'certainty_level': result.certainty_level,
                'model_predictions': result.model_predictions,
                'feature_importance': result.feature_importance,
                'mode': 'Ultra-Avancé'
            })
        else:
            score, label = detector.predict(img_path)
            return jsonify({
                'score': float(score),
                'prediction': label,
                'mode': 'Legacy'
            })
            
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        return jsonify({'error': f'Erreur prédiction: {str(e)}'}), 500


@app.route('/api/upload', methods=['POST'])
def upload():
    """Upload et analyse d'image ultra-avancée"""
    try:
        # Vérifications initiales
        if not os.path.exists(MODEL_PATH):
            return jsonify({'error': 'Modèle non trouvé. Veuillez entraîner le modèle d\'abord.'}), 400
        
        if 'image' not in request.files:
            return jsonify({'error': 'Aucun fichier uploadé. Veuillez sélectionner une image.'}), 400
        
        img_file = request.files['image']
        
        if img_file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
            
        if not allowed_file(img_file.filename):
            return jsonify({'error': 'Type de fichier invalide. Utilisez JPG, PNG ou WebP.'}), 400
        
        # Sauvegarde temporaire sécurisée
        filename = secure_filename(img_file.filename)
        temp_dir = tempfile.mkdtemp()
        img_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{filename}")
        
        img_file.save(img_path)
        
        # Validation de l'image
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            # Tentative de nettoyage avec PIL
            try:
                from PIL import Image
                pil_img = Image.open(img_path)
                pil_img = pil_img.convert('RGB')
                clean_path = os.path.join(temp_dir, f"clean_{uuid.uuid4()}.jpg")
                pil_img.save(clean_path, 'JPEG', quality=95)
                img_path = clean_path
                
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError("Image corrompue après nettoyage")
                    
            except Exception as img_error:
                shutil.rmtree(temp_dir)
                return jsonify({'error': f'Image corrompue: {str(img_error)}'}), 400
        
        # Validation des dimensions
        if img.shape[0] < 32 or img.shape[1] < 32:
            shutil.rmtree(temp_dir)
            return jsonify({'error': 'Image trop petite. Minimum: 32x32 pixels'}), 400
            
        if img.shape[0] > 4096 or img.shape[1] > 4096:
            shutil.rmtree(temp_dir)
            return jsonify({'error': 'Image trop grande. Maximum: 4096x4096 pixels'}), 400
        
        # Chargement du détecteur
        global detector
        if detector is None:
            logger.info("Chargement du modèle...")
            if ULTRA_MODE:
                detector = UltraEnhancedAIDetector()
                detector.load_ultra_model(MODEL_PATH)
            else:
                detector = AdaptiveAIDetector(MODEL_PATH)
            logger.info("Modèle chargé avec succès")
        
        # Analyse
        start_time = time.time()
        
        if ULTRA_MODE:
            result = detector.predict_ultra(img_path)
            response = {
                'score': result.score,
                'prediction': result.prediction,
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'image_quality': result.image_quality,
                'certainty_level': result.certainty_level,
                'model_predictions': result.model_predictions,
                'feature_importance': result.feature_importance,
                'anomaly_score': result.anomaly_score,
                'status': 'success',
                'mode': 'Ultra-Avancé v4.0',
                'algorithm_details': {
                    'features_extracted': len(result.features),
                    'models_used': list(result.model_predictions.keys()),
                    'ensemble_prediction': result.score
                }
            }
        else:
            score, label = detector.predict(img_path)
            response = {
                'score': float(score),
                'prediction': label,
                'processing_time': time.time() - start_time,
                'status': 'success',
                'mode': 'Legacy v3.0'
            }
        
        # Nettoyage
        shutil.rmtree(temp_dir)
        
        logger.info(f"Analyse terminée: {response['prediction']} (score: {response['score']:.4f})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erreur upload: {e}")
        if 'temp_dir' in locals():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        return jsonify({'error': f'Erreur analyse: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """État de santé de l'API"""
    return jsonify({
        'status': 'healthy',
        'mode': 'Ultra-Avancé v4.0' if ULTRA_MODE else 'Legacy v3.0',
        'model_loaded': detector is not None,
        'model_exists': os.path.exists(MODEL_PATH),
        'model_path': MODEL_PATH
    })


@app.route('/api/info', methods=['GET'])
def info():
    """Informations sur le système"""
    try:
        model_info = {}
        if detector and ULTRA_MODE:
            model_info = {
                'training_history': getattr(detector, 'training_history', {}),
                'config': getattr(detector, 'config', {}),
                'models_available': list(getattr(detector, 'models', {}).keys())
            }
        
        return jsonify({
            'system': 'AI Image Detector Ultra',
            'version': '4.0' if ULTRA_MODE else '3.0',
            'mode': 'Ultra-Avancé' if ULTRA_MODE else 'Legacy',
            'features': {
                'wavelets': ULTRA_MODE,
                'ensemble_models': ULTRA_MODE,
                'gabor_filters': ULTRA_MODE,
                'advanced_features': ULTRA_MODE
            },
            'model_info': model_info
        })
    except Exception as e:
        return jsonify({'error': f'Erreur info: {str(e)}'}), 500


# Routes pour servir le frontend
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_frontend_routes(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    print("🚀 Starting Ultra AI Image Detector Server...")
    print(f"🔧 Mode: {'Ultra-Avancé v4.0' if ULTRA_MODE else 'Legacy v3.0'}")
    print(f"🌐 Backend API: http://localhost:8000/api")
    print(f"🎨 Frontend: http://localhost:8000")
    
    # Tentative de chargement du modèle existant
    if os.path.exists(MODEL_PATH):
        try:
            print("📁 Chargement du modèle pré-entraîné...")
            if ULTRA_MODE:
                detector = UltraEnhancedAIDetector()
                detector.load_ultra_model(MODEL_PATH)
            else:
                detector = AdaptiveAIDetector(MODEL_PATH)
            print("✅ Modèle chargé avec succès!")
        except Exception as e:
            print(f"⚠️ Erreur chargement modèle: {e}")
            print("🔧 Le modèle sera rechargé lors de la première prédiction")
    else:
        print("ℹ️ Aucun modèle pré-entraîné trouvé. Entraînez le modèle d'abord.")
    
    app.run(host='0.0.0.0', port=8000, debug=True)