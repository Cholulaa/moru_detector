"""
Advanced Model Training System for AI Image Detection
====================================================

Système d'entraînement avancé avec:
- Ensemble de modèles (RF, XGBoost, SVM)
- Cross-validation et hyperparameter tuning
- Feature selection automatique
- Monitoring et métriques détaillées
- Sauvegarde/chargement optimisés

Author: Cholulaa
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import json
import time
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns


class AdvancedModelTrainer:
    """Système d'entraînement avancé pour la détection d'images IA"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logging()
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.ensemble_model = None
        self.training_history = {}
        self.feature_importance = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Configuration du logging avancé"""
        logger = logging.getLogger('model_trainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / 'training.log')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def train_ensemble_model(self, X: np.ndarray, y: np.ndarray, 
                           feature_names: List[str] = None) -> Dict:
        """Entraînement complet de l'ensemble de modèles"""
        start_time = time.time()
        self.logger.info("🚀 Début de l'entraînement de l'ensemble de modèles")
        
        # Validation des données
        X, y = self._validate_data(X, y)
        
        # Division train/test stratifiée
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.logger.info(f"📊 Données: {len(X_train)} train, {len(X_test)} test")
        self.logger.info(f"📈 Distribution: {np.bincount(y_train)} train, {np.bincount(y_test)} test")
        
        # Préparation des données
        X_train_scaled, X_test_scaled = self._prepare_data(X_train, X_test)
        
        # Sélection de features
        if self.config.get('feature_selection', {}).get('enabled', True):
            X_train_selected, X_test_selected, selected_features = self._select_features(
                X_train_scaled, y_train, X_test_scaled, feature_names
            )
        else:
            X_train_selected, X_test_selected = X_train_scaled, X_test_scaled
            selected_features = feature_names if feature_names else list(range(X.shape[1]))
        
        # Entraînement des modèles individuels
        individual_results = self._train_individual_models(
            X_train_selected, y_train, X_test_selected, y_test
        )
        
        # Construction de l'ensemble
        ensemble_results = self._build_ensemble_model(
            X_train_selected, y_train, X_test_selected, y_test
        )
        
        # Optimisation du seuil
        optimal_threshold = self._optimize_threshold(
            X_test_selected, y_test, self.ensemble_model
        )
        
        # Calcul des métriques finales
        final_metrics = self._calculate_final_metrics(
            X_test_selected, y_test, optimal_threshold
        )
        
        # Analyse d'importance des features
        self._analyze_feature_importance(selected_features)
        
        # Sauvegarde des résultats
        training_time = time.time() - start_time
        results = {
            'individual_models': individual_results,
            'ensemble_results': ensemble_results,
            'final_metrics': final_metrics,
            'optimal_threshold': optimal_threshold,
            'selected_features': selected_features,
            'training_time': training_time,
            'feature_importance': self.feature_importance
        }
        
        self._save_training_report(results)
        
        self.logger.info(f"✅ Entraînement terminé en {training_time:.2f}s")
        self.logger.info(f"🎯 AUC final: {final_metrics['auc']:.4f}")
        self.logger.info(f"🎯 F1-Score final: {final_metrics['f1_score']:.4f}")
        
        return results
    
    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validation et nettoyage des données"""
        self.logger.info("🔍 Validation des données...")
        
        # Vérification des dimensions
        if X.shape[0] != y.shape[0]:
            raise ValueError("X et y doivent avoir le même nombre d'échantillons")
        
        # Détection des valeurs manquantes
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.sum() > 0:
            self.logger.warning(f"⚠️ {nan_mask.sum()} échantillons avec NaN supprimés")
            X = X[~nan_mask]
            y = y[~nan_mask]
        
        # Détection des valeurs infinies
        inf_mask = np.isinf(X).any(axis=1)
        if inf_mask.sum() > 0:
            self.logger.warning(f"⚠️ {inf_mask.sum()} échantillons avec Inf supprimés")
            X = X[~inf_mask]
            y = y[~inf_mask]
        
        # Vérification de l'équilibre des classes
        class_counts = np.bincount(y)
        minority_ratio = min(class_counts) / max(class_counts)
        
        if minority_ratio < 0.1:
            self.logger.warning(f"⚠️ Classes déséquilibrées: ratio {minority_ratio:.3f}")
        
        self.logger.info(f"✅ Données validées: {X.shape[0]} échantillons, {X.shape[1]} features")
        
        return X, y
    
    def _prepare_data(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Préparation et normalisation des données"""
        self.logger.info("🔧 Préparation des données...")
        
        # Choix du scaler basé sur la configuration
        scaler_type = self.config.get('preprocessing', {}).get('scaler', 'standard')
        
        if scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        # Ajustement et transformation
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Sauvegarde du scaler
        self.scalers['main'] = scaler
        
        self.logger.info(f"✅ Normalisation {scaler_type} appliquée")
        
        return X_train_scaled, X_test_scaled