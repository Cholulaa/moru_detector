#!/usr/bin/env python3
"""
Test des imports pour le détecteur ultra-avancé
==============================================
"""

import sys
import os

print("🔧 Test des imports du détecteur ultra-avancé...")

# Ajouter le répertoire courant au path
sys.path.append('.')

# Test des imports de base
try:
    import numpy as np
    print("✅ NumPy: OK")
except ImportError as e:
    print(f"❌ NumPy: {e}")

try:
    import cv2
    print("✅ OpenCV: OK")
except ImportError as e:
    print(f"❌ OpenCV: {e}")

try:
    import sklearn
    print("✅ Scikit-learn: OK")
except ImportError as e:
    print(f"❌ Scikit-learn: {e}")

try:
    import scipy
    print("✅ SciPy: OK")
except ImportError as e:
    print(f"❌ SciPy: {e}")

# Test des imports optionnels (peuvent échouer)
try:
    import xgboost
    print("✅ XGBoost: OK")
except ImportError as e:
    print(f"⚠️ XGBoost: {e} (optionnel)")

try:
    import lightgbm
    print("✅ LightGBM: OK")
except ImportError as e:
    print(f"⚠️ LightGBM: {e} (optionnel)")

try:
    import pywt
    print("✅ PyWavelets: OK")
except ImportError as e:
    print(f"⚠️ PyWavelets: {e} (optionnel)")

try:
    import skimage
    print("✅ Scikit-image: OK")
except ImportError as e:
    print(f"⚠️ Scikit-image: {e} (optionnel)")

try:
    import yaml
    print("✅ PyYAML: OK")
except ImportError as e:
    print(f"❌ PyYAML: {e} (requis)")

# Test des imports du détecteur
try:
    from detector.ultra_feature_extractors import SpectralFeatureExtractor
    print("✅ Ultra Feature Extractors: OK")
except ImportError as e:
    print(f"❌ Ultra Feature Extractors: {e}")

try:
    from detector.ultra_enhanced_detector import UltraEnhancedAIDetector
    print("✅ Ultra Enhanced Detector: OK")
    
    # Test de création
    detector = UltraEnhancedAIDetector()
    print("✅ Création détecteur: OK")
    
except ImportError as e:
    print(f"❌ Ultra Enhanced Detector: {e}")
except Exception as e:
    print(f"❌ Erreur création détecteur: {e}")

print("\n🎉 Test d'imports terminé!")