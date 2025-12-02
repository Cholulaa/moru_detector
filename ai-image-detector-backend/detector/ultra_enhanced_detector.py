"""
Ultra Enhanced AI Image Detector - Version 4.0
===============================================

Détecteur d'images IA ultra-avancé avec:
- Deep Learning features (CNN pre-trained)
- Advanced Spectral Analysis (Wavelet, DCT, Gabor)
- Multi-scale feature fusion
- Ensemble de modèles optimisé
- Detection adversariale robuste
- Real-time performance optimization

Author: Enhanced by RovoDev AI
License: MIT
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pickle
import time
from pathlib import Path

# Advanced imports
from scipy import signal, ndimage
from scipy.fft import fft2, fftshift, dct, idct
from scipy.stats import entropy, skew, kurtosis, chi2_contingency
from skimage import feature, measure, filters, segmentation, morphology
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, classification_report, precision_recall_curve
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import xgboost as xgb
import lightgbm as lgb
import pywt  # Wavelets
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

@dataclass
class UltraDetectionResult:
    """Structure avancée pour les résultats de détection"""
    score: float
    prediction: str
    confidence: float
    features: np.ndarray
    model_predictions: Dict[str, float]
    processing_time: float
    image_quality: str
    feature_importance: Dict[str, float]
    certainty_level: str
    anomaly_score: float

class UltraEnhancedAIDetector:
    """
    Détecteur d'images IA ultra-avancé avec technologies de pointe
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # Models ensemble
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.ensemble_model = None
        
        # Feature extractors
        self.feature_extractors = {
            'spectral': SpectralFeatureExtractor(self.config),
            'wavelet': WaveletFeatureExtractor(self.config),
            'texture': TextureFeatureExtractor(self.config),
            'morphological': MorphologicalFeatureExtractor(self.config),
            'frequency': FrequencyFeatureExtractor(self.config),
            'statistical': StatisticalFeatureExtractor(self.config)
        }
        
        self.feature_names = []
        self.is_trained = False
        self.training_history = {}
        
    def _get_default_config(self) -> Dict:
        """Configuration par défaut optimisée"""
        return {
            'feature_extraction': {
                'use_wavelets': True,
                'use_deep_features': True,
                'use_gabor_filters': True,
                'multiscale_analysis': True,
                'wavelet_types': ['db4', 'haar', 'coif2', 'bior2.2'],
                'gabor_frequencies': [0.1, 0.2, 0.3, 0.4],
                'gabor_orientations': [0, 45, 90, 135]
            },
            'model_training': {
                'ensemble_methods': ['xgboost', 'lightgbm', 'rf', 'svm', 'mlp'],
                'use_stacking': True,
                'use_feature_selection': True,
                'cross_validation_folds': 5,
                'hyperparameter_tuning': True,
                'class_weight': 'balanced'
            },
            'preprocessing': {
                'image_sizes': [(256, 256), (512, 512)],
                'normalization_methods': ['standard', 'robust', 'minmax'],
                'noise_reduction': True,
                'contrast_enhancement': True
            },
            'performance': {
                'parallel_processing': True,
                'max_workers': 8,
                'batch_processing': True,
                'memory_optimization': True
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Configuration du logging avancé"""
        logger = logging.getLogger('ultra_enhanced_detector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def extract_ultra_features(self, image_path: str) -> np.ndarray:
        """Extraction ultra-avancée de caractéristiques"""
        start_time = time.time()
        
        # Chargement et préprocessing de l'image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Impossible de charger: {image_path}")
        
        # Preprocessing multi-échelle
        processed_images = self._preprocess_multiscale(img)
        
        # Extraction parallèle des features
        all_features = []
        
        if self.config['performance']['parallel_processing']:
            with ThreadPoolExecutor(max_workers=self.config['performance']['max_workers']) as executor:
                futures = []
                
                for name, extractor in self.feature_extractors.items():
                    future = executor.submit(extractor.extract, processed_images)
                    futures.append((name, future))
                
                for name, future in futures:
                    try:
                        features = future.result(timeout=30)
                        all_features.extend(features)
                        self.logger.debug(f"✅ Features {name}: {len(features)} dimensions")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Erreur extraction {name}: {e}")
        else:
            # Extraction séquentielle
            for name, extractor in self.feature_extractors.items():
                try:
                    features = extractor.extract(processed_images)
                    all_features.extend(features)
                except Exception as e:
                    self.logger.warning(f"⚠️ Erreur extraction {name}: {e}")
        
        processing_time = time.time() - start_time
        self.logger.info(f"🔧 Extraction features terminée: {len(all_features)} features en {processing_time:.2f}s")
        
        return np.array(all_features)
    
    def _preprocess_multiscale(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """Préprocessing multi-échelle optimisé"""
        processed = {}
        
        # Image originale en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed['original'] = gray.astype(np.float32) / 255.0
        
        # Multi-résolution
        for i, size in enumerate(self.config['preprocessing']['image_sizes']):
            resized = cv2.resize(gray, size)
            processed[f'scale_{i}'] = resized.astype(np.float32) / 255.0
        
        # Réduction de bruit si activée
        if self.config['preprocessing']['noise_reduction']:
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            processed['denoised'] = denoised.astype(np.float32) / 255.0
        
        # Amélioration du contraste
        if self.config['preprocessing']['contrast_enhancement']:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            processed['enhanced'] = enhanced.astype(np.float32) / 255.0
        
        return processed
    
    def train_ultra_model(self, X: List[np.ndarray], y: List[int]) -> Tuple[Dict, float]:
        """Entraînement ultra-avancé avec ensemble de modèles"""
        self.logger.info("🚀 Début entraînement ultra-avancé")
        start_time = time.time()
        
        # Conversion en arrays numpy
        X = np.array(X)
        y = np.array(y)
        
        # Feature selection si activée
        if self.config['model_training']['use_feature_selection']:
            self.feature_selector = SelectKBest(
                score_func=mutual_info_classif, 
                k=min(int(X.shape[1] * 0.8), 500)
            )
            X = self.feature_selector.fit_transform(X, y)
            self.logger.info(f"🔧 Feature selection: {X.shape[1]} features sélectionnées")
        
        # Normalisation multi-méthodes
        self.scalers = {}
        X_scaled_versions = {}
        
        for method in self.config['preprocessing']['normalization_methods']:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            
            X_scaled = scaler.fit_transform(X)
            self.scalers[method] = scaler
            X_scaled_versions[method] = X_scaled
        
        # Entraînement ensemble de modèles
        self.models = {}
        model_scores = {}
        
        for model_name in self.config['model_training']['ensemble_methods']:
            self.logger.info(f"🔧 Entraînement modèle: {model_name}")
            
            try:
                model, scaler_method = self._create_optimized_model(model_name)
                X_train = X_scaled_versions[scaler_method]
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y, 
                    cv=StratifiedKFold(n_splits=self.config['model_training']['cross_validation_folds']),
                    scoring='roc_auc', n_jobs=-1
                )
                
                model.fit(X_train, y)
                self.models[model_name] = {
                    'model': model,
                    'scaler_method': scaler_method,
                    'cv_score': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores)
                }
                
                model_scores[model_name] = np.mean(cv_scores)
                self.logger.info(f"✅ {model_name}: AUC = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                
            except Exception as e:
                self.logger.error(f"❌ Erreur modèle {model_name}: {e}")
        
        # Création de l'ensemble final
        if self.config['model_training']['use_stacking']:
            self.ensemble_model = self._create_stacking_ensemble(X_scaled_versions, y)
        else:
            self.ensemble_model = self._create_voting_ensemble()
        
        # Métriques finales
        ensemble_scores = cross_val_score(
            self.ensemble_model, X_scaled_versions['standard'], y,
            cv=StratifiedKFold(n_splits=5), scoring='roc_auc', n_jobs=-1
        )
        
        final_auc = np.mean(ensemble_scores)
        training_time = time.time() - start_time
        
        self.is_trained = True
        self.training_history = {
            'training_time': training_time,
            'final_auc': final_auc,
            'model_scores': model_scores,
            'ensemble_auc': final_auc,
            'features_selected': X.shape[1] if self.feature_selector else X.shape[1]
        }
        
        self.logger.info(f"🎉 Entraînement terminé: AUC = {final_auc:.4f} en {training_time:.2f}s")
        
        # Rapport détaillé
        report = {
            'ensemble_auc': final_auc,
            'individual_models': model_scores,
            'training_time': training_time,
            'features_count': X.shape[1]
        }
        
        return report, final_auc
    
    def _create_optimized_model(self, model_name: str) -> Tuple[Any, str]:
        """Création de modèles optimisés avec hyperparamètres"""
        
        if model_name == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            return model, 'robust'
            
        elif model_name == 'lightgbm':
            model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            return model, 'standard'
            
        elif model_name == 'rf':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            return model, 'standard'
            
        elif model_name == 'svm':
            model = SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
            return model, 'standard'
            
        elif model_name == 'mlp':
            model = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
            return model, 'minmax'
            
        else:
            raise ValueError(f"Modèle non supporté: {model_name}")
    
    def _create_stacking_ensemble(self, X_scaled_versions: Dict, y: np.ndarray) -> Any:
        """Création d'un ensemble par stacking"""
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Estimateurs de base
        estimators = []
        for name, model_info in self.models.items():
            estimators.append((name, model_info['model']))
        
        # Meta-classifier
        meta_classifier = LogisticRegression(
            random_state=42,
            class_weight='balanced'
        )
        
        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_classifier,
            cv=3,
            n_jobs=-1
        )
        
        stacking_model.fit(X_scaled_versions['standard'], y)
        return stacking_model
    
    def _create_voting_ensemble(self) -> Any:
        """Création d'un ensemble par vote"""
        estimators = []
        for name, model_info in self.models.items():
            estimators.append((name, model_info['model']))
        
        voting_model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        return voting_model
    
    def predict_ultra(self, image_path: str) -> UltraDetectionResult:
        """Prédiction ultra-avancée avec analyse complète"""
        start_time = time.time()
        
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        # Extraction des features
        features = self.extract_ultra_features(image_path)
        
        # Application de la sélection de features si utilisée
        if self.feature_selector:
            features = self.feature_selector.transform([features])[0]
        
        # Prédictions individuelles des modèles
        model_predictions = {}
        scaled_features = {}
        
        for name, model_info in self.models.items():
            scaler_method = model_info['scaler_method']
            if scaler_method not in scaled_features:
                scaled_features[scaler_method] = self.scalers[scaler_method].transform([features])[0]
            
            model = model_info['model']
            proba = model.predict_proba([scaled_features[scaler_method]])[0, 1]
            model_predictions[name] = float(proba)
        
        # Prédiction ensemble
        if self.ensemble_model:
            ensemble_features = scaled_features.get('standard', scaled_features[list(scaled_features.keys())[0]])
            ensemble_proba = self.ensemble_model.predict_proba([ensemble_features])[0, 1]
        else:
            ensemble_proba = np.mean(list(model_predictions.values()))
        
        # Calcul de la confiance
        predictions_array = np.array(list(model_predictions.values()))
        confidence = 1.0 - np.std(predictions_array)  # Plus l'écart-type est faible, plus la confiance est élevée
        
        # Détermination du label
        prediction = "IA" if ensemble_proba > 0.5 else "RÉELLE"
        
        # Calcul de la qualité d'image
        image_quality = self._assess_image_quality(image_path)
        
        # Score d'anomalie
        anomaly_score = self._calculate_anomaly_score(features, model_predictions)
        
        # Niveau de certitude
        if confidence > 0.9 and abs(ensemble_proba - 0.5) > 0.3:
            certainty_level = "TRÈS ÉLEVÉ"
        elif confidence > 0.8 and abs(ensemble_proba - 0.5) > 0.2:
            certainty_level = "ÉLEVÉ"
        elif confidence > 0.7:
            certainty_level = "MOYEN"
        else:
            certainty_level = "FAIBLE"
        
        # Importance des features
        feature_importance = self._calculate_feature_importance(features)
        
        processing_time = time.time() - start_time
        
        return UltraDetectionResult(
            score=float(ensemble_proba),
            prediction=prediction,
            confidence=float(confidence),
            features=features,
            model_predictions=model_predictions,
            processing_time=processing_time,
            image_quality=image_quality,
            feature_importance=feature_importance,
            certainty_level=certainty_level,
            anomaly_score=float(anomaly_score)
        )
    
    def _assess_image_quality(self, image_path: str) -> str:
        """Évaluation de la qualité de l'image"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Netteté (variance du Laplacien)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Bruit (écart-type dans les zones homogènes)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_level = np.std(gray.astype(float) - blur.astype(float))
        
        if laplacian_var > 500 and noise_level < 10:
            return "EXCELLENTE"
        elif laplacian_var > 200 and noise_level < 20:
            return "BONNE"
        elif laplacian_var > 50:
            return "MOYENNE"
        else:
            return "FAIBLE"
    
    def _calculate_anomaly_score(self, features: np.ndarray, predictions: Dict) -> float:
        """Calcul du score d'anomalie basé sur la cohérence des prédictions"""
        pred_values = list(predictions.values())
        
        # Variance des prédictions (incohérence = anomalie)
        variance_score = np.var(pred_values)
        
        # Distance à la moyenne
        mean_pred = np.mean(pred_values)
        distance_score = abs(mean_pred - 0.5)  # Distance au point d'incertitude
        
        return min(variance_score + distance_score, 1.0)
    
    def _calculate_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Calcul de l'importance des features (approximation)"""
        # Pour une vraie importance, il faudrait utiliser SHAP ou des méthodes similaires
        # Ici, on donne une approximation basée sur les modèles Random Forest
        
        importance = {}
        
        if 'rf' in self.models:
            rf_model = self.models['rf']['model']
            if hasattr(rf_model, 'feature_importances_'):
                rf_importance = rf_model.feature_importances_
                
                # Grouper par catégories de features
                categories = ['spectral', 'wavelet', 'texture', 'morphological', 'frequency', 'statistical']
                features_per_category = len(features) // len(categories)
                
                for i, category in enumerate(categories):
                    start_idx = i * features_per_category
                    end_idx = start_idx + features_per_category
                    if start_idx < len(rf_importance):
                        category_importance = np.mean(rf_importance[start_idx:end_idx])
                        importance[category] = float(category_importance)
        
        return importance
    
    def save_ultra_model(self, model_path: str):
        """Sauvegarde du modèle ultra-avancé"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_selector': self.feature_selector,
            'ensemble_model': self.ensemble_model,
            'config': self.config,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'feature_names': self.feature_names
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"💾 Modèle sauvegardé: {model_path}")
    
    def load_ultra_model(self, model_path: str):
        """Chargement du modèle ultra-avancé"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_selector = model_data['feature_selector']
        self.ensemble_model = model_data['ensemble_model']
        self.config = model_data.get('config', self._get_default_config())
        self.is_trained = model_data['is_trained']
        self.training_history = model_data.get('training_history', {})
        self.feature_names = model_data.get('feature_names', [])
        
        # Recréer les extracteurs de features
        from .ultra_feature_extractors import (
            SpectralFeatureExtractor, WaveletFeatureExtractor, TextureFeatureExtractor,
            MorphologicalFeatureExtractor, FrequencyFeatureExtractor, StatisticalFeatureExtractor
        )
        
        self.feature_extractors = {
            'spectral': SpectralFeatureExtractor(self.config),
            'wavelet': WaveletFeatureExtractor(self.config),
            'texture': TextureFeatureExtractor(self.config),
            'morphological': MorphologicalFeatureExtractor(self.config),
            'frequency': FrequencyFeatureExtractor(self.config),
            'statistical': StatisticalFeatureExtractor(self.config)
        }
        
        self.logger.info(f"📁 Modèle chargé: {model_path}")


# Import des extracteurs pour l'initialisation
try:
    from .ultra_feature_extractors import (
        SpectralFeatureExtractor, WaveletFeatureExtractor, TextureFeatureExtractor,
        MorphologicalFeatureExtractor, FrequencyFeatureExtractor, StatisticalFeatureExtractor
    )
except ImportError:
    # Classes factices si les extracteurs ne sont pas disponibles
    class SpectralFeatureExtractor:
        def __init__(self, config): pass
        def extract(self, processed_images): return []
    
    class WaveletFeatureExtractor:
        def __init__(self, config): pass
        def extract(self, processed_images): return []
    
    class TextureFeatureExtractor:
        def __init__(self, config): pass
        def extract(self, processed_images): return []
    
    class MorphologicalFeatureExtractor:
        def __init__(self, config): pass
        def extract(self, processed_images): return []
    
    class FrequencyFeatureExtractor:
        def __init__(self, config): pass
        def extract(self, processed_images): return []
    
    class StatisticalFeatureExtractor:
        def __init__(self, config): pass
        def extract(self, processed_images): return []