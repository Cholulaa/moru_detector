#!/usr/bin/env python3
"""
Ultra Advanced AI Image Detector Training Script
===============================================

Script d'entraînement ultra-avancé pour le détecteur d'images IA v4.0
Utilise les nouvelles technologies et algorithmes pour une précision maximale.

Usage:
    python train_ultra_model.py --real_dir dataset/real --ai_dir dataset/ai_generated
    python train_ultra_model.py --config ultra_config.yaml
    python train_ultra_model.py --auto_optimize

Author: Enhanced by RovoDev AI
License: MIT
"""

import argparse
import sys
import yaml
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
from datetime import datetime
import json
import os

# Import des modules ultra-avancés
try:
    from detector.ultra_enhanced_detector import UltraEnhancedAIDetector
    from detector.ultra_feature_extractors import *
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Assurez-vous que tous les modules sont installés")
    sys.exit(1)


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Configuration du système de logging ultra-avancé"""
    # Créer le dossier de logs
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'logs/ultra_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger('ultra_trainer')


def load_ultra_config(config_path: str = None) -> Dict:
    """Chargement de la configuration ultra-avancée"""
    default_config = {
        'feature_extraction': {
            'use_wavelets': True,
            'use_deep_features': False,  # Désactivé pour l'instant
            'use_gabor_filters': True,
            'multiscale_analysis': True,
            'wavelet_types': ['db4', 'haar', 'coif2'],
            'gabor_frequencies': [0.1, 0.2, 0.3, 0.4],
            'gabor_orientations': [0, 45, 90, 135]
        },
        'model_training': {
            'ensemble_methods': ['xgboost', 'lightgbm', 'rf', 'svm'],  # Retirer mlp pour la vitesse
            'use_stacking': True,
            'use_feature_selection': True,
            'cross_validation_folds': 5,
            'hyperparameter_tuning': False,  # Désactivé pour la vitesse
            'class_weight': 'balanced'
        },
        'preprocessing': {
            'image_sizes': [(256, 256), (512, 512)],
            'normalization_methods': ['standard', 'robust'],
            'noise_reduction': True,
            'contrast_enhancement': True
        },
        'performance': {
            'parallel_processing': True,
            'max_workers': min(8, os.cpu_count() or 1),
            'batch_processing': True,
            'memory_optimization': True
        }
    }
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            # Merger les configurations
            for key, value in user_config.items():
                if key in default_config and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
            logging.info(f"✅ Configuration chargée: {config_path}")
        except Exception as e:
            logging.warning(f"⚠️ Erreur de chargement de config: {e}, utilisation de la config par défaut")
    
    return default_config


def collect_image_paths(directory: str, extensions: List[str] = None) -> List[Path]:
    """Collection optimisée des chemins d'images"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Répertoire non trouvé: {directory}")
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(directory.rglob(f'*{ext}'))
        image_paths.extend(directory.rglob(f'*{ext.upper()}'))
    
    # Filtrer les fichiers trop petits ou corrompus
    valid_paths = []
    for path in image_paths:
        if path.stat().st_size > 1024:  # Au moins 1KB
            valid_paths.append(path)
    
    return valid_paths


def extract_features_parallel_ultra(image_paths: List[Path], 
                                   detector: UltraEnhancedAIDetector,
                                   max_workers: int = None) -> Tuple[List[np.ndarray], List[str]]:
    """Extraction parallèle ultra-optimisée des features"""
    logger = logging.getLogger('feature_extraction')
    
    if max_workers is None:
        max_workers = min(8, len(image_paths), os.cpu_count() or 1)
    
    logger.info(f"🔧 Extraction de features ultra-avancée avec {max_workers} workers")
    
    def extract_single_image(image_path: Path) -> Tuple[np.ndarray, str]:
        try:
            features = detector.extract_ultra_features(str(image_path))
            return features, str(image_path)
        except Exception as e:
            logger.warning(f"⚠️ Erreur extraction {image_path}: {e}")
            return None, str(image_path)
    
    all_features = []
    processed_paths = []
    failed_paths = []
    
    # Traitement par batches pour optimiser la mémoire
    batch_size = 100
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    with tqdm(total=len(image_paths), desc="Extraction features") as pbar:
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(extract_single_image, path): path for path in batch_paths}
                
                for future in as_completed(futures):
                    features, path = future.result()
                    if features is not None:
                        all_features.append(features)
                        processed_paths.append(path)
                    else:
                        failed_paths.append(path)
                    
                    pbar.update(1)
    
    logger.info(f"✅ Features extraites: {len(all_features)} succès, {len(failed_paths)} échecs")
    if failed_paths:
        logger.warning(f"⚠️ Images échouées: {failed_paths[:5]}..." if len(failed_paths) > 5 else f"⚠️ Images échouées: {failed_paths}")
    
    return all_features, processed_paths


def train_ultra_model_optimized(real_dir: str, ai_dir: str, config: Dict, output_path: str = None) -> Dict:
    """Entraînement ultra-optimisé du modèle"""
    logger = logging.getLogger('ultra_training')
    
    start_time = time.time()
    logger.info("🚀 Début de l'entraînement ultra-avancé")
    
    # Collection des images
    logger.info("📁 Collection des images...")
    real_paths = collect_image_paths(real_dir)
    ai_paths = collect_image_paths(ai_dir)
    
    logger.info(f"📊 Dataset: {len(real_paths)} images réelles, {len(ai_paths)} images IA")
    
    if len(real_paths) == 0 or len(ai_paths) == 0:
        raise ValueError("Aucune image trouvée dans un des dossiers")
    
    # Équilibrage du dataset si nécessaire
    min_count = min(len(real_paths), len(ai_paths))
    if min_count < 100:
        logger.warning(f"⚠️ Dataset très petit: {min_count} images par classe")
    
    # Limiter pour éviter les déséquilibres extrêmes
    max_per_class = min(len(real_paths), len(ai_paths), 5000)  # Limiter à 5000 par classe max
    real_paths = real_paths[:max_per_class]
    ai_paths = ai_paths[:max_per_class]
    
    logger.info(f"📊 Dataset équilibré: {len(real_paths)} images réelles, {len(ai_paths)} images IA")
    
    # Initialisation du détecteur
    detector = UltraEnhancedAIDetector(config)
    
    # Extraction des features
    logger.info("🔧 Extraction des features...")
    real_features, real_processed = extract_features_parallel_ultra(
        real_paths, detector, config['performance']['max_workers']
    )
    ai_features, ai_processed = extract_features_parallel_ultra(
        ai_paths, detector, config['performance']['max_workers']
    )
    
    # Préparation des données d'entraînement
    X = real_features + ai_features
    y = [0] * len(real_features) + [1] * len(ai_features)  # 0 = réelle, 1 = IA
    
    logger.info(f"📊 Données d'entraînement: {len(X)} échantillons, {len(X[0]) if X else 0} features")
    
    if len(X) < 20:
        raise ValueError("Pas assez d'échantillons pour l'entraînement (minimum 20)")
    
    # Entraînement du modèle
    logger.info("🎯 Entraînement du modèle ultra-avancé...")
    training_report, final_auc = detector.train_ultra_model(X, y)
    
    # Sauvegarde du modèle
    if output_path is None:
        output_path = f"models/ultra_ai_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    detector.save_ultra_model(output_path)
    
    # Rapport final
    training_time = time.time() - start_time
    
    final_report = {
        'training_time': training_time,
        'dataset_stats': {
            'real_images': len(real_features),
            'ai_images': len(ai_features),
            'total_samples': len(X),
            'features_count': len(X[0]) if X else 0
        },
        'model_performance': training_report,
        'model_path': output_path,
        'config_used': config
    }
    
    # Sauvegarde du rapport
    report_path = output_path.replace('.pkl', '_report.json')
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    logger.info("🎉 Entraînement terminé avec succès!")
    logger.info(f"📊 AUC final: {final_auc:.4f}")
    logger.info(f"⏱️ Temps d'entraînement: {training_time:.1f}s")
    logger.info(f"💾 Modèle sauvegardé: {output_path}")
    logger.info(f"📄 Rapport sauvegardé: {report_path}")
    
    return final_report


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Entraînement ultra-avancé du détecteur d\'images IA')
    parser.add_argument('--real_dir', type=str, help='Dossier des images réelles')
    parser.add_argument('--ai_dir', type=str, help='Dossier des images IA')
    parser.add_argument('--config', type=str, help='Fichier de configuration YAML')
    parser.add_argument('--output', type=str, help='Chemin de sauvegarde du modèle')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--auto_optimize', action='store_true', help='Optimisation automatique des hyperparamètres')
    
    args = parser.parse_args()
    
    # Configuration du logging
    logger = setup_logging(args.log_level)
    
    try:
        # Chargement de la configuration
        config = load_ultra_config(args.config)
        
        # Optimisation automatique si demandée
        if args.auto_optimize:
            logger.info("🔧 Optimisation automatique activée")
            config['model_training']['hyperparameter_tuning'] = True
            config['performance']['max_workers'] = min(16, os.cpu_count() or 1)
        
        # Vérification des dossiers
        if not args.real_dir or not args.ai_dir:
            # Utiliser les dossiers par défaut si disponibles
            real_dir = 'dataset/real_images'
            ai_dir = 'dataset/ai_generated'
            
            if not (Path(real_dir).exists() and Path(ai_dir).exists()):
                logger.error("❌ Dossiers d'images non spécifiés ou non trouvés")
                logger.error("Utilisation: python train_ultra_model.py --real_dir path/to/real --ai_dir path/to/ai")
                sys.exit(1)
        else:
            real_dir = args.real_dir
            ai_dir = args.ai_dir
        
        # Entraînement
        report = train_ultra_model_optimized(real_dir, ai_dir, config, args.output)
        
        # Affichage du résumé
        print("\n" + "="*80)
        print("🎉 ENTRAÎNEMENT ULTRA-AVANCÉ TERMINÉ AVEC SUCCÈS!")
        print("="*80)
        print(f"📊 AUC Final: {report['model_performance']['ensemble_auc']:.4f}")
        print(f"⏱️ Temps d'entraînement: {report['training_time']:.1f}s")
        print(f"📈 Échantillons: {report['dataset_stats']['total_samples']}")
        print(f"🔧 Features: {report['dataset_stats']['features_count']}")
        print(f"💾 Modèle: {report['model_path']}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ Erreur critique: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()