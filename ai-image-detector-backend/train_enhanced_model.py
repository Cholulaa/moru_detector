#!/usr/bin/env python3
"""
Enhanced AI Image Detector Training Script
==========================================

Script d'entraînement avancé pour le détecteur d'images IA v3.0
Utilise les nouvelles features et l'ensemble de modèles pour une précision maximale.

Usage:
    python train_enhanced_model.py --real_dir dataset/real --ai_dir dataset/ai_generated
    python train_enhanced_model.py --config custom_config.yaml
    python train_enhanced_model.py --auto_download_datasets

Author: Cholulaa
License: MIT
"""

import argparse
import sys
import yaml
import numpy as np
from pathlib import Path
from typing import List, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
from datetime import datetime

# Import des modules améliorés
from detector.enhanced_detector import EnhancedAIDetector
from detector.feature_extractor import AdvancedFeatureExtractor
from detector.model_trainer import AdvancedModelTrainer


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Configuration du système de logging"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/training_enhanced.log')
        ]
    )
    return logging.getLogger('enhanced_trainer')


def load_config(config_path: str = 'config.yaml') -> dict:
    """Chargement de la configuration"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"✅ Configuration chargée: {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"❌ Fichier de configuration non trouvé: {config_path}")
        sys.exit(1)


def collect_image_paths(directory: str, extensions: List[str] = None) -> List[Path]:
    """Collection des chemins d'images dans un répertoire"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
    
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Répertoire non trouvé: {directory}")
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(directory.rglob(f'*{ext}'))
        image_paths.extend(directory.rglob(f'*{ext.upper()}'))
    
    return image_paths


def extract_features_parallel(image_paths: List[Path], 
                            feature_extractor: AdvancedFeatureExtractor,
                            n_workers: int = None) -> Tuple[np.ndarray, List[str]]:
    """Extraction parallèle des features"""
    logger = logging.getLogger('feature_extraction')
    
    if n_workers is None:
        n_workers = min(8, len(image_paths))
    
    logger.info(f"🔧 Extraction de features avec {n_workers} workers")
    
    def extract_single_image(image_path: Path) -> np.ndarray:
        try:
            features = feature_extractor.extract_all_features(str(image_path))
            return features
        except Exception as e:
            logger.warning(f"⚠️ Erreur extraction {image_path}: {e}")
            return None
    
    all_features = []
    failed_images = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Soumission des tâches
        future_to_path = {
            executor.submit(extract_single_image, path): path 
            for path in image_paths
        }
        
        # Collecte des résultats avec barre de progression
        for future in tqdm(as_completed(future_to_path), 
                          total=len(image_paths), 
                          desc="Extraction features"):
            path = future_to_path[future]
            try:
                features = future.result()
                if features is not None:
                    all_features.append(features)
                else:
                    failed_images.append(path)
            except Exception as e:
                logger.warning(f"⚠️ Erreur lors du traitement de {path}: {e}")
                failed_images.append(path)
    
    logger.info(f"✅ Features extraites: {len(all_features)}/{len(image_paths)}")
    if failed_images:
        logger.warning(f"⚠️ Images échouées: {len(failed_images)}")
    
    return np.array(all_features), failed_images


def prepare_dataset(real_dir: str, ai_dir: str, 
                   feature_extractor: AdvancedFeatureExtractor,
                   config: dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Préparation complète du dataset"""
    logger = logging.getLogger('dataset_preparation')
    logger.info("📊 Préparation du dataset...")
    
    # Collecte des images
    real_images = collect_image_paths(real_dir)
    ai_images = collect_image_paths(ai_dir)
    
    logger.info(f"📁 Images réelles: {len(real_images)}")
    logger.info(f"📁 Images IA: {len(ai_images)}")
    
    # Vérification de l'équilibre
    if len(real_images) == 0 or len(ai_images) == 0:
        raise ValueError("Au moins une catégorie d'images est vide")
    
    # Limitation du nombre d'images si configuré
    max_images_per_class = config.get('dataset', {}).get('max_images_per_class')
    if max_images_per_class:
        real_images = real_images[:max_images_per_class]
        ai_images = ai_images[:max_images_per_class]
        logger.info(f"📊 Limitation appliquée: {len(real_images)} + {len(ai_images)} images")
    
    # Extraction des features
    n_workers = config.get('performance', {}).get('n_jobs', -1)
    if n_workers == -1:
        import os
        n_workers = os.cpu_count()
    
    logger.info("🔧 Extraction features images réelles...")
    real_features, real_failed = extract_features_parallel(
        real_images, feature_extractor, n_workers
    )
    
    logger.info("🔧 Extraction features images IA...")
    ai_features, ai_failed = extract_features_parallel(
        ai_images, feature_extractor, n_workers
    )
    
    # Construction du dataset final
    X = np.vstack([real_features, ai_features])
    y = np.hstack([
        np.zeros(len(real_features)),  # 0 = réel
        np.ones(len(ai_features))      # 1 = IA
    ])
    
    # Noms des features pour l'interprétabilité
    feature_names = feature_extractor.feature_names[:X.shape[1]]
    
    logger.info(f"✅ Dataset final: {X.shape[0]} échantillons, {X.shape[1]} features")
    logger.info(f"📊 Distribution: {np.bincount(y.astype(int))}")
    
    return X, y, feature_names


def download_sample_datasets(target_dir: str = 'dataset') -> Tuple[str, str]:
    """Téléchargement automatique de datasets d'exemple"""
    logger = logging.getLogger('dataset_download')
    logger.info("⬇️ Téléchargement de datasets d'exemple...")
    
    # URLs d'exemple de datasets (à adapter selon vos besoins)
    datasets_info = {
        'real_images': {
            'url': 'https://example.com/real_images.zip',
            'description': 'Images réelles haute qualité'
        },
        'ai_generated': {
            'url': 'https://example.com/ai_images.zip', 
            'description': 'Images générées par différents modèles IA'
        }
    }
    
    target_dir = Path(target_dir)
    target_dir.mkdir(exist_ok=True)
    
    real_dir = target_dir / 'real_images'
    ai_dir = target_dir / 'ai_generated'
    
    # Note: Implémentation simplifiée - dans un vrai projet,
    # vous devriez télécharger et extraire les datasets
    logger.warning("⚠️ Auto-download non implémenté dans cette démo")
    logger.info("📋 Créez manuellement les dossiers:")
    logger.info(f"   - {real_dir}: Images authentiques")
    logger.info(f"   - {ai_dir}: Images générées par IA")
    
    real_dir.mkdir(exist_ok=True)
    ai_dir.mkdir(exist_ok=True)
    
    return str(real_dir), str(ai_dir)


def create_training_report(results: dict, config: dict, 
                         real_dir: str, ai_dir: str) -> None:
    """Création d'un rapport d'entraînement détaillé"""
    logger = logging.getLogger('reporting')
    
    report_dir = Path('results')
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f'training_report_{timestamp}.json'
    
    # Préparation du rapport
    report = {
        'timestamp': timestamp,
        'config': config,
        'dataset': {
            'real_images_dir': real_dir,
            'ai_images_dir': ai_dir,
            'total_samples': results.get('total_samples', 0),
            'features_count': results.get('features_count', 0)
        },
        'training_results': results,
        'model_performance': {
            'final_auc': results.get('final_metrics', {}).get('auc', 0),
            'final_f1': results.get('final_metrics', {}).get('f1_score', 0),
            'optimal_threshold': results.get('optimal_threshold', 0.5),
            'training_time': results.get('training_time', 0)
        },
        'feature_analysis': {
            'selected_features': results.get('selected_features', []),
            'feature_importance': results.get('feature_importance', {})
        }
    }
    
    # Sauvegarde
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"📄 Rapport sauvegardé: {report_path}")
    
    # Affichage du résumé
    print("\n" + "="*80)
    print("🎯 RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("="*80)
    print(f"⏱️  Temps d'entraînement: {results.get('training_time', 0):.2f}s")
    print(f"📊 Échantillons: {report['dataset']['total_samples']}")
    print(f"🔧 Features: {report['dataset']['features_count']}")
    print(f"🎯 AUC final: {report['model_performance']['final_auc']:.4f}")
    print(f"🎯 F1-Score final: {report['model_performance']['final_f1']:.4f}")
    print(f"🎚️  Seuil optimal: {report['model_performance']['optimal_threshold']:.4f}")
    print("="*80)


def main():
    """Fonction principale d'entraînement"""
    parser = argparse.ArgumentParser(
        description='Enhanced AI Image Detector Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python train_enhanced_model.py --real_dir dataset/real --ai_dir dataset/ai
  python train_enhanced_model.py --config custom_config.yaml
  python train_enhanced_model.py --auto_download_datasets
        """
    )
    
    parser.add_argument('--real_dir', type=str, 
                       help='Répertoire des images réelles')
    parser.add_argument('--ai_dir', type=str,
                       help='Répertoire des images générées par IA')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Fichier de configuration YAML')
    parser.add_argument('--auto_download_datasets', action='store_true',
                       help='Téléchargement automatique de datasets d\'exemple')
    parser.add_argument('--output_model', type=str, 
                       default='models/enhanced_ai_detector_v3.pkl',
                       help='Chemin de sauvegarde du modèle')
    
    args = parser.parse_args()
    
    # Configuration du logging
    Path('logs').mkdir(exist_ok=True)
    logger = setup_logging()
    
    logger.info("🚀 Démarrage de l'entraînement Enhanced AI Detector v3.0")
    
    # Chargement de la configuration
    config = load_config(args.config)
    
    # Gestion des datasets
    if args.auto_download_datasets:
        real_dir, ai_dir = download_sample_datasets()
    else:
        real_dir = args.real_dir
        ai_dir = args.ai_dir
    
    if not real_dir or not ai_dir:
        logger.error("❌ Spécifiez --real_dir et --ai_dir ou utilisez --auto_download_datasets")
        sys.exit(1)
    
    try:
        # Initialisation des composants
        logger.info("🔧 Initialisation des composants...")
        feature_extractor = AdvancedFeatureExtractor(config['feature_extraction'])
        model_trainer = AdvancedModelTrainer(config)
        
        # Préparation du dataset
        X, y, feature_names = prepare_dataset(
            real_dir, ai_dir, feature_extractor, config
        )
        
        # Entraînement du modèle
        logger.info("🎯 Démarrage de l'entraînement...")
        results = model_trainer.train_ensemble_model(X, y, feature_names)
        
        # Ajout d'informations au résultat
        results['total_samples'] = len(X)
        results['features_count'] = X.shape[1]
        
        # Sauvegarde du modèle
        logger.info("💾 Sauvegarde du modèle...")
        Path(args.output_model).parent.mkdir(exist_ok=True)
        
        model_data = {
            'ensemble_model': model_trainer.ensemble_model,
            'scalers': model_trainer.scalers,
            'feature_selector': model_trainer.feature_selector,
            'config': config,
            'feature_names': feature_names,
            'results': results
        }
        
        import joblib
        joblib.dump(model_data, args.output_model)
        logger.info(f"✅ Modèle sauvegardé: {args.output_model}")
        
        # Création du rapport
        create_training_report(results, config, real_dir, ai_dir)
        
        logger.info("🎉 Entraînement terminé avec succès!")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'entraînement: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()