#!/usr/bin/env python3
"""
Setup Script pour le Détecteur d'Images IA Ultra-Avancé v4.0
============================================================

Script d'installation et configuration automatique pour:
- Installation des dépendances
- Configuration de l'environnement
- Test du système
- Migration depuis l'ancien système

Author: Enhanced by RovoDev AI
License: MIT
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path
import json
import logging
from datetime import datetime

def setup_logging():
    """Configuration du logging pour l'installation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('setup_ultra.log')
        ]
    )
    return logging.getLogger('setup_ultra')

def check_python_version():
    """Vérification de la version Python"""
    logger = logging.getLogger('setup_ultra')
    
    if sys.version_info < (3, 7):
        logger.error("❌ Python 3.7+ requis. Version actuelle: %s", sys.version)
        return False
    
    logger.info("✅ Version Python OK: %s", sys.version.split()[0])
    return True

def install_requirements():
    """Installation des dépendances"""
    logger = logging.getLogger('setup_ultra')
    
    logger.info("📦 Installation des dépendances...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True, capture_output=True, text=True)
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True, text=True)
        
        logger.info("✅ Dépendances installées avec succès")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error("❌ Erreur installation: %s", e.stderr)
        return False

def create_directory_structure():
    """Création de la structure de dossiers"""
    logger = logging.getLogger('setup_ultra')
    
    directories = [
        'models',
        'logs',
        'dataset/real_images',
        'dataset/ai_generated',
        'test_images',
        'results',
        'config'
    ]
    
    logger.info("📁 Création de la structure de dossiers...")
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Créé: {directory}")
    
    return True

def create_config_files():
    """Création des fichiers de configuration"""
    logger = logging.getLogger('setup_ultra')
    
    # Configuration ultra-avancée
    ultra_config = {
        'feature_extraction': {
            'use_wavelets': True,
            'use_gabor_filters': True,
            'multiscale_analysis': True,
            'wavelet_types': ['db4', 'haar', 'coif2', 'bior2.2'],
            'gabor_frequencies': [0.1, 0.2, 0.3, 0.4],
            'gabor_orientations': [0, 45, 90, 135]
        },
        'model_training': {
            'ensemble_methods': ['xgboost', 'lightgbm', 'rf', 'svm'],
            'use_stacking': True,
            'use_feature_selection': True,
            'cross_validation_folds': 5,
            'hyperparameter_tuning': False,
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
            'max_workers': min(8, os.cpu_count() or 1),
            'batch_processing': True,
            'memory_optimization': True
        }
    }
    
    # Sauvegarde de la configuration
    with open('config/ultra_config.yaml', 'w') as f:
        import yaml
        yaml.dump(ultra_config, f, default_flow_style=False, indent=2)
    
    logger.info("✅ Configuration ultra-avancée créée: config/ultra_config.yaml")
    
    # Configuration rapide pour tests
    quick_config = ultra_config.copy()
    quick_config['model_training']['ensemble_methods'] = ['rf', 'xgboost']
    quick_config['preprocessing']['image_sizes'] = [(256, 256)]
    quick_config['performance']['max_workers'] = 2
    
    with open('config/quick_test_config.yaml', 'w') as f:
        yaml.dump(quick_config, f, default_flow_style=False, indent=2)
    
    logger.info("✅ Configuration test rapide créée: config/quick_test_config.yaml")
    
    return True

def test_imports():
    """Test des imports critiques"""
    logger = logging.getLogger('setup_ultra')
    
    logger.info("🧪 Test des imports...")
    
    imports_to_test = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('sklearn', 'Scikit-learn'),
        ('xgboost', 'XGBoost'),
        ('lightgbm', 'LightGBM'),
        ('pywt', 'PyWavelets'),
        ('skimage', 'Scikit-image'),
        ('flask', 'Flask'),
        ('yaml', 'PyYAML')
    ]
    
    failed_imports = []
    
    for module, name in imports_to_test:
        try:
            __import__(module)
            logger.info(f"✅ {name}")
        except ImportError:
            logger.error(f"❌ {name}")
            failed_imports.append(name)
    
    if failed_imports:
        logger.error("❌ Imports échoués: %s", ', '.join(failed_imports))
        return False
    
    logger.info("✅ Tous les imports réussis")
    return True

def test_detector_creation():
    """Test de création du détecteur ultra-avancé"""
    logger = logging.getLogger('setup_ultra')
    
    logger.info("🧪 Test de création du détecteur...")
    
    try:
        # Test d'import et création
        sys.path.append('.')
        from detector.ultra_enhanced_detector import UltraEnhancedAIDetector
        
        detector = UltraEnhancedAIDetector()
        logger.info("✅ Détecteur ultra-avancé créé avec succès")
        
        # Test d'extraction de features (si image de test disponible)
        test_images = list(Path('test_images').glob('*.jpg')) + list(Path('test_images').glob('*.png'))
        if test_images:
            test_image = test_images[0]
            features = detector.extract_ultra_features(str(test_image))
            logger.info(f"✅ Test extraction features: {len(features)} features extraites")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur test détecteur: {e}")
        return False

def migrate_old_model():
    """Migration depuis l'ancien système si disponible"""
    logger = logging.getLogger('setup_ultra')
    
    old_model_path = "models/ai_detector_model.pkl"
    
    if Path(old_model_path).exists():
        logger.info("🔄 Ancien modèle détecté, sauvegarde...")
        backup_path = f"models/ai_detector_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        shutil.copy2(old_model_path, backup_path)
        logger.info(f"✅ Ancien modèle sauvegardé: {backup_path}")
        
        # Créer un fichier d'information
        info = {
            'backup_date': datetime.now().isoformat(),
            'original_path': old_model_path,
            'backup_path': backup_path,
            'note': 'Ancien modèle sauvegardé avant mise à jour ultra-avancée'
        }
        
        with open('models/migration_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        return True
    
    logger.info("ℹ️ Aucun ancien modèle trouvé")
    return True

def create_start_scripts():
    """Création de scripts de démarrage"""
    logger = logging.getLogger('setup_ultra')
    
    # Script de démarrage ultra
    ultra_start_script = """#!/bin/bash
# Script de démarrage pour le détecteur Ultra-Avancé v4.0

echo "🚀 Démarrage du Détecteur d'Images IA Ultra-Avancé v4.0"
echo "=================================================="

# Activation de l'environnement virtuel si disponible
if [ -d "venv" ]; then
    echo "📦 Activation de l'environnement virtuel..."
    source venv/bin/activate
fi

# Démarrage du serveur
echo "🌐 Démarrage du serveur..."
python app_ultra.py

echo "✅ Serveur arrêté"
"""
    
    with open('start_ultra.sh', 'w') as f:
        f.write(ultra_start_script)
    
    # Rendre exécutable sur Unix
    if os.name != 'nt':
        os.chmod('start_ultra.sh', 0o755)
    
    logger.info("✅ Script de démarrage créé: start_ultra.sh")
    
    # Script de test rapide
    test_script = """#!/usr/bin/env python3
# Test rapide du système

import sys
import os
sys.path.append('.')

try:
    from detector.ultra_enhanced_detector import UltraEnhancedAIDetector
    print("✅ Import détecteur ultra: OK")
    
    detector = UltraEnhancedAIDetector()
    print("✅ Création détecteur: OK")
    
    print("🎉 Système ultra-avancé opérationnel!")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    sys.exit(1)
"""
    
    with open('test_system.py', 'w') as f:
        f.write(test_script)
    
    logger.info("✅ Script de test créé: test_system.py")
    
    return True

def print_summary():
    """Affichage du résumé d'installation"""
    print("\n" + "="*80)
    print("🎉 INSTALLATION DU DÉTECTEUR ULTRA-AVANCÉ v4.0 TERMINÉE!")
    print("="*80)
    print()
    print("📁 Structure créée:")
    print("   ├── models/                 # Modèles entraînés")
    print("   ├── dataset/               # Données d'entraînement")
    print("   ├── logs/                  # Journaux système")
    print("   ├── config/                # Configurations")
    print("   └── test_images/           # Images de test")
    print()
    print("🚀 ÉTAPES SUIVANTES:")
    print("   1. Placez vos images dans dataset/real_images/ et dataset/ai_generated/")
    print("   2. Entraînez le modèle: python train_ultra_model.py --real_dir dataset/real_images --ai_dir dataset/ai_generated")
    print("   3. Démarrez le serveur: python app_ultra.py ou ./start_ultra.sh")
    print("   4. Accédez à l'interface: http://localhost:8000")
    print()
    print("🔧 COMMANDES UTILES:")
    print("   • Test système: python test_system.py")
    print("   • Entraînement rapide: python train_ultra_model.py --config config/quick_test_config.yaml")
    print("   • Logs: tail -f logs/ultra_training_*.log")
    print()
    print("📚 NOUVEAUTÉS v4.0:")
    print("   ✨ Extraction de features multi-échelles (Wavelets, Gabor)")
    print("   ✨ Ensemble de modèles avancés (XGBoost, LightGBM, RF, SVM)")
    print("   ✨ Analyse morphologique et texturale")
    print("   ✨ Détection d'anomalies et évaluation de confiance")
    print("   ✨ Optimisation des performances et traitement parallèle")
    print("="*80)

def main():
    """Fonction principale d'installation"""
    print("🚀 Installation du Détecteur d'Images IA Ultra-Avancé v4.0")
    print("=" * 60)
    
    logger = setup_logging()
    logger.info("Début de l'installation ultra-avancée")
    
    steps = [
        ("Vérification Python", check_python_version),
        ("Installation dépendances", install_requirements),
        ("Création structure", create_directory_structure),
        ("Configuration", create_config_files),
        ("Test imports", test_imports),
        ("Test détecteur", test_detector_creation),
        ("Migration", migrate_old_model),
        ("Scripts", create_start_scripts)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        logger.info(f"📋 {step_name}...")
        try:
            if step_func():
                logger.info(f"✅ {step_name}: OK")
            else:
                logger.error(f"❌ {step_name}: ÉCHEC")
                failed_steps.append(step_name)
        except Exception as e:
            logger.error(f"❌ {step_name}: ERREUR - {e}")
            failed_steps.append(step_name)
    
    if failed_steps:
        logger.error("❌ Installation échouée. Étapes en erreur: %s", ', '.join(failed_steps))
        print(f"\n❌ Installation échouée. Vérifiez le fichier setup_ultra.log")
        return False
    else:
        logger.info("✅ Installation terminée avec succès")
        print_summary()
        return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)