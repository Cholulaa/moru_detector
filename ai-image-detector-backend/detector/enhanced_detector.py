"""
Enhanced AI Image Detector - Version 3.0
==========================================

Détecteur d'images IA de nouvelle génération avec:
- Ensemble de modèles ML avancés
- Features engineering scientifique
- Architecture robuste et extensible
- Logging et monitoring complets
- Résistance aux attaques adversariales

Author: Cholulaa
License: MIT
"""

import numpy as np
import cv2
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import signal, ndimage
from scipy.fft import fft2, fftshift
from scipy.stats import entropy, skew, kurtosis
from skimage import feature, measure, filters, segmentation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc, classification_report
import xgboost as xgb
import pickle
import json
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DetectionResult:
    """Structure pour les résultats de détection"""
    score: float
    prediction: str
    confidence: float
    features: np.ndarray
    model_predictions: Dict[str, float]
    processing_time: float
    image_quality: str


class EnhancedAIDetector:
    """
    Détecteur d'images IA avancé avec ensemble de modèles
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.models = {}
        self.scalers = {}
        self.feature_importance = None
        self.is_trained = False
        
        # Configuration des modèles
        self.model_config = {
            'random_forest': {
                'n_estimators': 300,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            },
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'logloss'
            },
            'gradient_boost': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42
            }
        }
        
        self.logger.info("Enhanced AI Detector initialisé")
    
    def load_model(self, model_path: str) -> bool:
        """Chargement d'un modèle pré-entraîné"""
        try:
            import joblib
            model_data = joblib.load(model_path)
            
            if isinstance(model_data, dict):
                # Nouveau format avec ensemble
                self.ensemble_model = model_data.get('ensemble_model')
                self.scalers = model_data.get('scalers', {})
                self.feature_selector = model_data.get('feature_selector')
                self.is_trained = True
                self.logger.info("✅ Modèle ensemble chargé")
            else:
                # Format legacy - adapter
                self.models['legacy'] = model_data
                self.is_trained = True
                self.logger.info("⚠️ Modèle legacy chargé")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement modèle: {e}")
            return False
    
    def predict(self, image_path: str) -> Tuple[float, str]:
        """Prédiction sur une image"""
        start_time = time.time()
        
        try:
            if not self.is_trained:
                raise ValueError("Modèle non entraîné")
            
            # Extraction des features
            if hasattr(self, 'feature_extractor'):
                features = self.feature_extractor.extract_all_features(image_path)
            else:
                # Fallback vers extraction basique
                features = self._extract_basic_features(image_path)
            
            features = features.reshape(1, -1)
            
            # Prédiction
            if self.ensemble_model:
                # Modèle ensemble
                if 'main' in self.scalers:
                    features_scaled = self.scalers['main'].transform(features)
                else:
                    features_scaled = features
                
                if self.feature_selector:
                    features_scaled = self.feature_selector.transform(features_scaled)
                
                score = self.ensemble_model.predict_proba(features_scaled)[0, 1]
            else:
                # Modèle legacy
                legacy_model = self.models.get('legacy')
                if legacy_model and hasattr(legacy_model, 'predict_proba'):
                    score = legacy_model.predict_proba(features)[0, 1]
                else:
                    score = 0.5  # Fallback
            
            # Classification
            label = self._classify_score(score)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Prédiction: {score:.4f} ({label}) en {processing_time:.3f}s")
            
            return float(score), label
            
        except Exception as e:
            self.logger.error(f"❌ Erreur prédiction: {e}")
            raise RuntimeError(f"Erreur lors de la prédiction: {e}")
    
    def _extract_basic_features(self, image_path: str) -> np.ndarray:
        """Extraction basique de features (fallback)"""
        import cv2
        from scipy.fft import fft2, fftshift
        
        # Chargement image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Impossible de charger: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_norm = gray.astype(np.float32) / 255.0
        
        # Features FFT basiques (compatibilité)
        h, w = gray_norm.shape
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            gray_norm = cv2.resize(gray_norm, (new_w, new_h))
        
        # FFT
        fft = fft2(gray_norm)
        fft_shifted = fftshift(fft)
        magnitude = np.abs(fft_shifted)
        
        # Features simples
        features = [
            np.mean(magnitude),
            np.std(magnitude),
            np.max(magnitude),
            np.min(magnitude),
            np.var(magnitude)
        ]
        
        # Padding pour avoir 35 features (compatibilité legacy)
        while len(features) < 35:
            features.append(0.0)
        
        return np.array(features[:35])
    
    def _classify_score(self, score: float) -> str:
        """Classification du score en label"""
        if score < 0.3:
            return "AUTHENTIQUE"
        elif score < 0.7:
            return "INCERTAIN"
        else:
            return "IA GÉNÉRÉE"
    
    def save_model(self, model_path: str) -> bool:
        """Sauvegarde du modèle"""
        try:
            import joblib
            
            model_data = {
                'ensemble_model': getattr(self, 'ensemble_model', None),
                'scalers': self.scalers,
                'feature_selector': getattr(self, 'feature_selector', None),
                'config': self.config,
                'is_trained': self.is_trained,
                'version': '3.0'
            }
            
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_data, model_path)
            
            self.logger.info(f"✅ Modèle sauvegardé: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde: {e}")
            return False
    
    def _setup_logging(self) -> logging.Logger:
        """Configuration du système de logging"""
        logger = logging.getLogger('ai_detector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Chargement de la configuration"""
        default_config = {
            'feature_extraction': {
                'use_deep_features': False,
                'use_texture_features': True,
                'use_color_features': True,
                'max_image_size': 2048
            },
            'model_ensemble': {
                'use_voting': True,
                'weights': [0.4, 0.3, 0.3],  # RF, XGB, GB
                'threshold_optimization': True
            },
            'quality_assessment': {
                'min_resolution': 64,
                'compression_check': True,
                'noise_analysis': True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config