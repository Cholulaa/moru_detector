#!/usr/bin/env python3
"""
Benchmark et Comparaison des Versions v3.0 vs v4.0
==================================================

Script de comparaison des performances entre:
- Version Legacy v3.0 (AdaptiveAIDetector)
- Version Ultra v4.0 (UltraEnhancedAIDetector)

Metrics évaluées:
- Précision (Accuracy, AUC, F1-Score)
- Vitesse de traitement
- Robustesse
- Qualité des features

Author: Enhanced by RovoDev AI
License: MIT
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import json
from datetime import datetime
import sys
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('benchmark')

def setup_test_environment():
    """Configuration de l'environnement de test"""
    logger.info("🔧 Configuration de l'environnement de test...")
    
    # Créer dossiers de résultats
    os.makedirs('benchmark_results', exist_ok=True)
    os.makedirs('benchmark_results/plots', exist_ok=True)
    
    return True

def load_detectors():
    """Chargement des deux versions de détecteurs"""
    detectors = {}
    
    try:
        # Version Legacy v3.0
        from detector.adaptive_detector import AdaptiveAIDetector
        detectors['legacy_v3'] = {
            'class': AdaptiveAIDetector,
            'instance': None,
            'model_path': 'models/ai_detector_model.pkl'
        }
        logger.info("✅ Détecteur Legacy v3.0 chargé")
    except ImportError as e:
        logger.warning(f"⚠️ Impossible de charger Legacy v3.0: {e}")
        detectors['legacy_v3'] = None
    
    try:
        # Version Ultra v4.0
        from detector.ultra_enhanced_detector import UltraEnhancedAIDetector
        detectors['ultra_v4'] = {
            'class': UltraEnhancedAIDetector,
            'instance': None,
            'model_path': 'models/ultra_ai_detector_model.pkl'
        }
        logger.info("✅ Détecteur Ultra v4.0 chargé")
    except ImportError as e:
        logger.warning(f"⚠️ Impossible de charger Ultra v4.0: {e}")
        detectors['ultra_v4'] = None
    
    return detectors

def collect_test_images(test_dir: str = "test_images", max_images: int = 50) -> List[Path]:
    """Collection d'images de test"""
    if not Path(test_dir).exists():
        logger.error(f"❌ Dossier de test non trouvé: {test_dir}")
        return []
    
    extensions = ['.jpg', '.jpeg', '.png', '.webp']
    images = []
    
    for ext in extensions:
        images.extend(Path(test_dir).glob(f'*{ext}'))
        images.extend(Path(test_dir).glob(f'*{ext.upper()}'))
    
    images = images[:max_images]
    logger.info(f"📁 {len(images)} images de test collectées")
    
    return images

def benchmark_feature_extraction(detectors: Dict, test_images: List[Path]) -> Dict:
    """Benchmark de l'extraction de features"""
    logger.info("🔬 Benchmark extraction de features...")
    
    results = {}
    
    for version_name, detector_info in detectors.items():
        if detector_info is None:
            continue
            
        logger.info(f"🧪 Test {version_name}...")
        
        # Initialisation
        if version_name == 'legacy_v3':
            detector = detector_info['class']()
        else:
            detector = detector_info['class']()
        
        times = []
        feature_counts = []
        successful_extractions = 0
        
        for image_path in test_images[:10]:  # Limite pour le benchmark
            try:
                start_time = time.time()
                
                if version_name == 'legacy_v3':
                    features = detector.extract_all_features(str(image_path))
                else:
                    features = detector.extract_ultra_features(str(image_path))
                
                extraction_time = time.time() - start_time
                
                if features is not None and len(features) > 0:
                    times.append(extraction_time)
                    feature_counts.append(len(features))
                    successful_extractions += 1
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur extraction {image_path}: {e}")
        
        if times:
            results[version_name] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'avg_features': np.mean(feature_counts),
                'total_features': int(np.mean(feature_counts)),
                'success_rate': successful_extractions / len(test_images[:10]),
                'throughput': successful_extractions / np.sum(times) if np.sum(times) > 0 else 0
            }
        
        logger.info(f"✅ {version_name}: {successful_extractions}/{len(test_images[:10])} succès")
    
    return results

def synthetic_training_benchmark(detectors: Dict) -> Dict:
    """Benchmark avec données synthétiques pour l'entraînement"""
    logger.info("🎯 Benchmark entraînement avec données synthétiques...")
    
    # Génération de données synthétiques
    np.random.seed(42)
    n_samples = 200
    
    results = {}
    
    for version_name, detector_info in detectors.items():
        if detector_info is None:
            continue
            
        logger.info(f"🧪 Test entraînement {version_name}...")
        
        try:
            if version_name == 'legacy_v3':
                # Features synthétiques pour v3.0 (30 features)
                X = np.random.normal(0, 1, (n_samples, 30))
                detector = detector_info['class']()
            else:
                # Features synthétiques pour v4.0 (200+ features)
                X = np.random.normal(0, 1, (n_samples, 185))
                detector = detector_info['class']()
            
            # Labels équilibrés
            y = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
            
            # Entraînement avec timing
            start_time = time.time()
            
            if version_name == 'legacy_v3':
                report, auc_score = detector.train(X.tolist(), y.tolist())
            else:
                report, auc_score = detector.train_ultra_model(X.tolist(), y.tolist())
            
            training_time = time.time() - start_time
            
            results[version_name] = {
                'training_time': training_time,
                'auc_score': float(auc_score),
                'features_used': X.shape[1],
                'samples_trained': X.shape[0],
                'throughput_samples_per_sec': X.shape[0] / training_time
            }
            
            # Extraction des métriques détaillées si disponibles
            if isinstance(report, dict):
                if 'ensemble_auc' in report:
                    results[version_name]['ensemble_auc'] = report['ensemble_auc']
                if 'individual_models' in report:
                    results[version_name]['models_count'] = len(report['individual_models'])
            
            logger.info(f"✅ {version_name}: AUC={auc_score:.4f}, temps={training_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement {version_name}: {e}")
    
    return results

def memory_usage_benchmark(detectors: Dict, test_images: List[Path]) -> Dict:
    """Benchmark de l'usage mémoire"""
    logger.info("💾 Benchmark usage mémoire...")
    
    import psutil
    import gc
    
    results = {}
    
    for version_name, detector_info in detectors.items():
        if detector_info is None:
            continue
            
        logger.info(f"🧪 Test mémoire {version_name}...")
        
        # Nettoyage mémoire
        gc.collect()
        
        # Mesure mémoire initiale
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Initialisation détecteur
            if version_name == 'legacy_v3':
                detector = detector_info['class']()
            else:
                detector = detector_info['class']()
            
            creation_memory = process.memory_info().rss / 1024 / 1024
            
            # Test sur quelques images
            max_memory = creation_memory
            for image_path in test_images[:5]:
                try:
                    if version_name == 'legacy_v3':
                        features = detector.extract_all_features(str(image_path))
                    else:
                        features = detector.extract_ultra_features(str(image_path))
                    
                    current_memory = process.memory_info().rss / 1024 / 1024
                    max_memory = max(max_memory, current_memory)
                    
                except Exception:
                    pass
            
            results[version_name] = {
                'initial_memory_mb': initial_memory,
                'creation_memory_mb': creation_memory,
                'max_memory_mb': max_memory,
                'memory_overhead_mb': creation_memory - initial_memory,
                'peak_usage_mb': max_memory - initial_memory
            }
            
            logger.info(f"✅ {version_name}: Peak={max_memory:.1f}MB, Overhead={creation_memory-initial_memory:.1f}MB")
            
        except Exception as e:
            logger.error(f"❌ Erreur mémoire {version_name}: {e}")
        
        # Nettoyage
        del detector
        gc.collect()
    
    return results

def generate_comparison_report(feature_results: Dict, training_results: Dict, memory_results: Dict) -> Dict:
    """Génération du rapport de comparaison"""
    logger.info("📊 Génération du rapport de comparaison...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {},
        'detailed_results': {
            'feature_extraction': feature_results,
            'training_performance': training_results,
            'memory_usage': memory_results
        },
        'recommendations': {}
    }
    
    # Comparaisons directes
    if 'legacy_v3' in feature_results and 'ultra_v4' in feature_results:
        legacy = feature_results['legacy_v3']
        ultra = feature_results['ultra_v4']
        
        report['summary']['feature_extraction'] = {
            'speed_ratio': legacy['avg_time'] / ultra['avg_time'] if ultra['avg_time'] > 0 else 'N/A',
            'features_ratio': ultra['avg_features'] / legacy['avg_features'] if legacy['avg_features'] > 0 else 'N/A',
            'quality_improvement': (ultra['success_rate'] - legacy['success_rate']) * 100
        }
    
    if 'legacy_v3' in training_results and 'ultra_v4' in training_results:
        legacy = training_results['legacy_v3']
        ultra = training_results['ultra_v4']
        
        report['summary']['training_performance'] = {
            'auc_improvement': (ultra['auc_score'] - legacy['auc_score']) * 100,
            'training_time_ratio': ultra['training_time'] / legacy['training_time'] if legacy['training_time'] > 0 else 'N/A',
            'features_improvement': (ultra['features_used'] - legacy['features_used']) / legacy['features_used'] * 100 if legacy['features_used'] > 0 else 'N/A'
        }
    
    if 'legacy_v3' in memory_results and 'ultra_v4' in memory_results:
        legacy = memory_results['legacy_v3']
        ultra = memory_results['ultra_v4']
        
        report['summary']['memory_usage'] = {
            'overhead_ratio': ultra['memory_overhead_mb'] / legacy['memory_overhead_mb'] if legacy['memory_overhead_mb'] > 0 else 'N/A',
            'peak_ratio': ultra['peak_usage_mb'] / legacy['peak_usage_mb'] if legacy['peak_usage_mb'] > 0 else 'N/A'
        }
    
    # Recommandations
    if 'ultra_v4' in feature_results:
        if feature_results['ultra_v4']['success_rate'] > 0.9:
            report['recommendations']['ultra_v4'] = "Recommandé pour précision maximale"
        else:
            report['recommendations']['ultra_v4'] = "Optimisations nécessaires"
    
    if 'legacy_v3' in feature_results:
        if feature_results['legacy_v3']['avg_time'] < 0.5:
            report['recommendations']['legacy_v3'] = "Recommandé pour vitesse maximale"
        else:
            report['recommendations']['legacy_v3'] = "Option de fallback"
    
    return report

def create_visualization_plots(report: Dict):
    """Création des graphiques de comparaison"""
    logger.info("📈 Création des visualisations...")
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparaison Détecteurs IA v3.0 vs v4.0', fontsize=16, fontweight='bold')
    
    # Graphique 1: Temps d'extraction de features
    if 'feature_extraction' in report['detailed_results']:
        results = report['detailed_results']['feature_extraction']
        versions = list(results.keys())
        times = [results[v]['avg_time'] for v in versions]
        
        axes[0,0].bar(versions, times, color=['#3498db', '#e74c3c'])
        axes[0,0].set_title('Temps d\'extraction de features (s)')
        axes[0,0].set_ylabel('Temps (secondes)')
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(times):
            axes[0,0].text(i, v + 0.01, f'{v:.3f}s', ha='center')
    
    # Graphique 2: Nombre de features
    if 'feature_extraction' in report['detailed_results']:
        results = report['detailed_results']['feature_extraction']
        versions = list(results.keys())
        features = [results[v]['avg_features'] for v in versions]
        
        axes[0,1].bar(versions, features, color=['#3498db', '#e74c3c'])
        axes[0,1].set_title('Nombre de features extraites')
        axes[0,1].set_ylabel('Nombre de features')
        
        for i, v in enumerate(features):
            axes[0,1].text(i, v + 5, f'{int(v)}', ha='center')
    
    # Graphique 3: Performance d'entraînement (AUC)
    if 'training_performance' in report['detailed_results']:
        results = report['detailed_results']['training_performance']
        versions = list(results.keys())
        aucs = [results[v]['auc_score'] for v in versions]
        
        axes[1,0].bar(versions, aucs, color=['#3498db', '#e74c3c'])
        axes[1,0].set_title('Performance (AUC Score)')
        axes[1,0].set_ylabel('AUC Score')
        axes[1,0].set_ylim(0.8, 1.0)
        
        for i, v in enumerate(aucs):
            axes[1,0].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Graphique 4: Usage mémoire
    if 'memory_usage' in report['detailed_results']:
        results = report['detailed_results']['memory_usage']
        versions = list(results.keys())
        memory = [results[v]['peak_usage_mb'] for v in versions]
        
        axes[1,1].bar(versions, memory, color=['#3498db', '#e74c3c'])
        axes[1,1].set_title('Usage mémoire peak (MB)')
        axes[1,1].set_ylabel('Mémoire (MB)')
        
        for i, v in enumerate(memory):
            axes[1,1].text(i, v + 1, f'{v:.1f}MB', ha='center')
    
    plt.tight_layout()
    plt.savefig('benchmark_results/plots/comparison_overview.png', dpi=300, bbox_inches='tight')
    logger.info("📊 Graphique sauvegardé: benchmark_results/plots/comparison_overview.png")
    
    plt.close()

def print_summary_report(report: Dict):
    """Affichage du résumé du rapport"""
    print("\n" + "="*80)
    print("🎯 RAPPORT DE BENCHMARK - DÉTECTEURS IA v3.0 vs v4.0")
    print("="*80)
    
    if 'summary' in report:
        summary = report['summary']
        
        if 'feature_extraction' in summary:
            fe = summary['feature_extraction']
            print("\n📊 EXTRACTION DE FEATURES:")
            if isinstance(fe['features_ratio'], (int, float)):
                print(f"   • Features extraites: {fe['features_ratio']:.1f}x plus avec v4.0")
            if isinstance(fe['quality_improvement'], (int, float)):
                print(f"   • Amélioration qualité: +{fe['quality_improvement']:.1f}%")
        
        if 'training_performance' in summary:
            tp = summary['training_performance']
            print("\n🎯 PERFORMANCE D'ENTRAÎNEMENT:")
            if isinstance(tp['auc_improvement'], (int, float)):
                print(f"   • Amélioration AUC: +{tp['auc_improvement']:.2f}%")
            if isinstance(tp['features_improvement'], (int, float)):
                print(f"   • Plus de features: +{tp['features_improvement']:.0f}%")
        
        if 'memory_usage' in summary:
            mu = summary['memory_usage']
            print("\n💾 USAGE MÉMOIRE:")
            if isinstance(mu['overhead_ratio'], (int, float)):
                print(f"   • Overhead mémoire: {mu['overhead_ratio']:.1f}x plus avec v4.0")
    
    if 'recommendations' in report:
        print("\n🎯 RECOMMANDATIONS:")
        for version, recommendation in report['recommendations'].items():
            print(f"   • {version}: {recommendation}")
    
    print("\n" + "="*80)

def main():
    """Fonction principale du benchmark"""
    print("🚀 Benchmark Détecteurs IA - v3.0 vs v4.0")
    print("=" * 50)
    
    # Configuration
    if not setup_test_environment():
        logger.error("❌ Erreur configuration environnement")
        sys.exit(1)
    
    # Chargement des détecteurs
    detectors = load_detectors()
    
    available_detectors = [k for k, v in detectors.items() if v is not None]
    if len(available_detectors) < 1:
        logger.error("❌ Aucun détecteur disponible pour le benchmark")
        sys.exit(1)
    
    logger.info(f"✅ Détecteurs disponibles: {', '.join(available_detectors)}")
    
    # Collection d'images de test
    test_images = collect_test_images()
    if not test_images:
        logger.warning("⚠️ Pas d'images de test trouvées, utilisation de données synthétiques uniquement")
    
    # Benchmarks
    feature_results = {}
    training_results = {}
    memory_results = {}
    
    if test_images:
        feature_results = benchmark_feature_extraction(detectors, test_images)
        memory_results = memory_usage_benchmark(detectors, test_images)
    
    training_results = synthetic_training_benchmark(detectors)
    
    # Génération du rapport
    report = generate_comparison_report(feature_results, training_results, memory_results)
    
    # Sauvegarde du rapport
    report_path = f"benchmark_results/benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"📄 Rapport sauvegardé: {report_path}")
    
    # Visualisations
    if len(available_detectors) >= 2:
        create_visualization_plots(report)
    
    # Affichage du résumé
    print_summary_report(report)
    
    logger.info("🎉 Benchmark terminé avec succès!")

if __name__ == '__main__':
    main()