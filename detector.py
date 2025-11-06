import numpy as np
import cv2
from scipy import signal
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from pathlib import Path
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import csv
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AdaptiveAIDetector:
    """
    Détecteur adaptatif d'images générées par IA avec multiprocessing
    """
    
    def __init__(self, model_path=None):
        self.weights = {
            'spectral_artifacts': 0.40,
            'frequency_convolution': 0.35,
            'phase_anomaly': 0.25
        }
        
        self.ml_model = None
        self.scaler = StandardScaler()
        self.optimal_threshold = 0.5
        self.is_trained = False
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def extract_all_features(self, image_path):
        """Extraction complète de toutes les caractéristiques fréquentielles"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Impossible de charger: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_norm = gray.astype(np.float32) / 255.0
        h, w = gray_norm.shape
        
        # FFT
        fft = fft2(gray_norm)
        fft_shifted = fftshift(fft)
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        magnitude_log = np.log(magnitude + 1e-10)
        
        # Extraction de toutes les features
        hf_features = self._extract_high_frequency_features(magnitude_log, phase, (h, w))
        psd_features = self._extract_psd_features(magnitude, (h, w))
        upsampling_features = self._extract_upsampling_features(magnitude, (h, w))
        conv_features = self._extract_frequency_convolution_features(fft_shifted, (h, w))
        phase_features = self._extract_phase_features(phase)
        global_features = self._extract_global_statistics(magnitude_log, phase)
        multiscale_features = self._extract_multiscale_features(gray_norm)
        
        all_features = np.concatenate([
            hf_features, psd_features, upsampling_features,
            conv_features, phase_features, global_features, multiscale_features
        ])
        
        return all_features
    
    def _extract_high_frequency_features(self, magnitude, phase, shape):
        """5 features haute fréquence"""
        h, w = shape
        center_h, center_w = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        radius = min(center_h, center_w) // 3
        mask_high = ((X - center_w)**2 + (Y - center_h)**2) > radius**2
        
        mag_hf = magnitude * mask_high
        phase_hf = phase * mask_high
        mag_norm = (mag_hf - np.mean(mag_hf)) / (np.std(mag_hf) + 1e-10)
        
        return np.array([
            np.sum(np.abs(mag_norm) > 2.5) / (h * w),
            np.std(mag_hf),
            np.mean(np.abs(mag_hf)),
            np.sum(np.abs(np.diff(phase_hf, axis=0)) > np.pi/2) / (h * w),
            np.max(mag_hf) - np.min(mag_hf)
        ])
    
    def _extract_psd_features(self, magnitude, shape):
        """5 features PSD"""
        h, w = shape
        psd = magnitude ** 2
        center_h, center_w = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        r = np.sqrt((X - center_w)**2 + (Y - center_h)**2)
        r_int = r.astype(int)
        max_radius = min(int(np.min([center_h, center_w])), 100)
        
        radial_profile = np.zeros(max_radius)
        for i in range(max_radius):
            mask = (r_int == i)
            if np.sum(mask) > 0:
                radial_profile[i] = np.mean(psd[mask])
        
        if len(radial_profile) > 10:
            gradient = np.gradient(radial_profile)
            return np.array([
                np.var(gradient),
                np.sum(np.abs(np.diff(gradient)) > np.std(gradient) * 2) / len(gradient),
                np.mean(radial_profile[:10]),
                np.mean(radial_profile[-10:]) if len(radial_profile) > 10 else 0,
                np.sum(psd) / (h * w)
            ])
        return np.array([0.0, 0.0, 0.0, 0.0, np.sum(psd) / (h * w)])
    
    def _extract_upsampling_features(self, magnitude, shape):
        """5 features upsampling"""
        h, w = shape
        freqs = [h//8, h//4, h//2]
        energies = []
        
        for freq in freqs:
            if freq < h and freq < w:
                window = 5
                fh = min(freq, h-window)
                fw = min(freq, w-window)
                local_energy = np.sum(magnitude[fh:fh+window, fw:fw+window] ** 2)
                energies.append(local_energy)
            else:
                energies.append(0)
        
        total_energy = np.sum(magnitude ** 2) + 1e-10
        return np.array([
            energies[0] / total_energy,
            energies[1] / total_energy,
            energies[2] / total_energy,
            sum(energies) / total_energy,
            np.std(energies)
        ])
    
    def _extract_frequency_convolution_features(self, fft_shifted, shape):
        """5 features convolution fréquentielle"""
        f_am = np.abs(fft_shifted)
        f_ph = np.angle(fft_shifted)
        kernel = np.ones((5, 5)) / 25
        
        f_am_log = np.log(f_am + 1e-10)
        f_am_conv = signal.convolve2d(f_am_log, kernel, mode='same', boundary='wrap')
        f_ph_conv = signal.convolve2d(f_ph, kernel, mode='same', boundary='wrap')
        
        return np.array([
            np.std(f_am_conv) / (np.mean(np.abs(f_am_conv)) + 1e-10),
            np.std(f_ph_conv) / (np.pi + 1e-10),
            np.var(f_am_conv),
            np.var(f_ph_conv),
            np.corrcoef(f_am_conv.flatten(), f_ph_conv.flatten())[0,1]
        ])
    
    def _extract_phase_features(self, phase):
        """5 features phase"""
        phase_grad_h = np.angle(np.exp(1j * np.diff(phase, axis=0)))
        phase_grad_w = np.angle(np.exp(1j * np.diff(phase, axis=1)))
        
        return np.array([
            np.std(phase_grad_h),
            np.std(phase_grad_w),
            np.mean(np.abs(phase_grad_h)),
            np.mean(np.abs(phase_grad_w)),
            np.var(phase)
        ])
    
    def _extract_global_statistics(self, magnitude, phase):
        """5 features statistiques globales"""
        return np.array([
            np.mean(magnitude),
            np.std(magnitude),
            np.mean(phase),
            np.std(phase),
            np.median(magnitude)
        ])
    
    def _extract_multiscale_features(self, image):
        """5 features multi-échelle"""
        scales = [1.0, 0.5, 0.25]
        energies = []
        
        for scale in scales:
            if scale < 1.0:
                h, w = image.shape
                new_h, new_w = int(h * scale), int(w * scale)
                img_scaled = cv2.resize(image, (new_w, new_h))
            else:
                img_scaled = image
            
            fft = np.abs(fft2(img_scaled))
            energy = np.sum(fft ** 2)
            energies.append(energy)
        
        return np.array([
            energies[0],
            energies[1] / (energies[0] + 1e-10),
            energies[2] / (energies[0] + 1e-10),
            np.std(energies),
            np.max(energies) / (np.min(energies) + 1e-10)
        ])
    
    def train(self, real_images_folder, fake_images_folder, test_size=0.2, 
              save_model_path="model_trained.pkl", n_workers=None):
        """Entraînement du modèle avec multiprocessing"""
        print("\n" + "="*70)
        print("ENTRAÎNEMENT DU DÉTECTEUR ADAPTATIF (MODE PARALLÈLE)")
        print("="*70)
        
        # Détermination du nombre de workers
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)  # Laisser 1 CPU libre
        
        print(f"\n⚡ Utilisation de {n_workers} processus parallèles")
        print(f"   (CPU disponibles: {cpu_count()})")
        
        real_folder = Path(real_images_folder)
        fake_folder = Path(fake_images_folder)
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Collecte des chemins d'images
        real_paths = [str(p) for p in real_folder.iterdir() if p.suffix.lower() in extensions]
        fake_paths = [str(p) for p in fake_folder.iterdir() if p.suffix.lower() in extensions]
        
        print(f"\n📁 Images trouvées:")
        print(f"   - Réelles: {len(real_paths)}")
        print(f"   - IA: {len(fake_paths)}")
        
        # Traitement parallèle des images RÉELLES
        print("\n[PHASE 1] Extraction parallèle des features - IMAGES RÉELLES")
        print("-" * 70)
        X_real = self._parallel_feature_extraction(real_paths, n_workers, "Réelles")
        
        # Traitement parallèle des images IA
        print("\n[PHASE 2] Extraction parallèle des features - IMAGES IA")
        print("-" * 70)
        X_fake = self._parallel_feature_extraction(fake_paths, n_workers, "IA")
        
        # Combinaison des datasets
        X = np.array(X_real + X_fake)
        y = np.array([0] * len(X_real) + [1] * len(X_fake))
        
        print("\n" + "="*70)
        print(f"DATASET COMPLET: {len(X)} images")
        print(f"  - Images RÉELLES: {len(X_real)}")
        print(f"  - Images IA: {len(X_fake)}")
        print("="*70)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\n[PHASE 3] Division Train/Test")
        print(f"  - Train: {len(X_train)} images ({np.sum(y_train==0)} réelles, {np.sum(y_train==1)} IA)")
        print(f"  - Test: {len(X_test)} images ({np.sum(y_test==0)} réelles, {np.sum(y_test==1)} IA)")
        
        # Normalisation
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entraînement
        print(f"\n[PHASE 4] Entraînement du modèle Random Forest...")
        self.ml_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.ml_model.fit(X_train_scaled, y_train)
        print("✓ Entraînement terminé")
        
        # Optimisation du seuil
        print(f"\n[PHASE 5] Optimisation du seuil de décision...")
        y_proba = self.ml_model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        optimal_idx = np.argmax(tpr - fpr)
        self.optimal_threshold = thresholds[optimal_idx]
        
        print(f"✓ Seuil optimal: {self.optimal_threshold:.3f}")
        print(f"✓ AUC-ROC: {roc_auc:.3f}")
        
        # Évaluation
        y_pred = (y_proba >= self.optimal_threshold).astype(int)
        
        print("\n" + "="*70)
        print("PERFORMANCE GLOBALE SUR LE TEST SET")
        print("="*70)
        print(classification_report(y_test, y_pred, target_names=['Réelle', 'IA']))
        
        cm = confusion_matrix(y_test, y_pred)
        print("\nMatrice de confusion:")
        print(f"                    Prédiction: RÉELLE  |  Prédiction: IA")
        print(f"  Vérité: RÉELLE    {cm[0,0]:>6}         |  {cm[0,1]:>6} (Faux Positifs)")
        print(f"  Vérité: IA        {cm[1,0]:>6} (Faux Négatifs) |  {cm[1,1]:>6}")
        
        # Analyse détaillée par catégorie
        real_indices = np.where(y_test == 0)[0]
        fake_indices = np.where(y_test == 1)[0]
        
        real_proba = y_proba[real_indices]
        fake_proba = y_proba[fake_indices]
        
        print("\n" + "-"*70)
        print("ANALYSE DÉTAILLÉE PAR CATÉGORIE")
        print("-"*70)
        print(f"\nImages RÉELLES (n={len(real_proba)}):")
        print(f"  Score moyen: {np.mean(real_proba):.3f}")
        print(f"  Écart-type: {np.std(real_proba):.3f}")
        print(f"  Min: {np.min(real_proba):.3f} | Max: {np.max(real_proba):.3f}")
        print(f"  Bien classées: {np.sum(real_proba < self.optimal_threshold)} / {len(real_proba)} ({np.sum(real_proba < self.optimal_threshold)/len(real_proba)*100:.1f}%)")
        
        print(f"\nImages IA (n={len(fake_proba)}):")
        print(f"  Score moyen: {np.mean(fake_proba):.3f}")
        print(f"  Écart-type: {np.std(fake_proba):.3f}")
        print(f"  Min: {np.min(fake_proba):.3f} | Max: {np.max(fake_proba):.3f}")
        print(f"  Bien classées: {np.sum(fake_proba >= self.optimal_threshold)} / {len(fake_proba)} ({np.sum(fake_proba >= self.optimal_threshold)/len(fake_proba)*100:.1f}%)")
        
        self.is_trained = True
        
        # Sauvegarde
        self.save_model(save_model_path)
        print(f"\n✓ Modèle sauvegardé: {save_model_path}")
        print("="*70 + "\n")
        
        return {
            'real_scores': real_proba,
            'fake_scores': fake_proba,
            'threshold': self.optimal_threshold,
            'auc': roc_auc
        }
    
    def _parallel_feature_extraction(self, image_paths, n_workers, label):
        """
        Extraction parallèle des features avec ProcessPoolExecutor
        """
        features_list = []
        failed_count = 0
        
        # Utilisation de ProcessPoolExecutor pour CPU-intensive tasks
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Soumission de tous les jobs
            future_to_path = {
                executor.submit(extract_features_worker, path): path 
                for path in image_paths
            }
            
            # Utilisation de tqdm pour une belle barre de progression
            with tqdm(total=len(image_paths), desc=f"  Processing {label}", 
                     unit="img", ncols=100) as pbar:
                
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        features = future.result()
                        if features is not None:
                            features_list.append(features)
                        else:
                            failed_count += 1
                    except Exception as e:
                        failed_count += 1
                        # print(f"\n  ✗ Erreur {Path(path).name}: {e}")
                    
                    pbar.update(1)
        
        print(f"  ✓ Traité: {len(features_list)} images | Échecs: {failed_count}")
        
        return features_list
    
    def predict(self, image_path, verbose=True):
        """Prédiction avec modèle entraîné"""
        features = self.extract_all_features(image_path)
        
        if self.is_trained and self.ml_model is not None:
            features_scaled = self.scaler.transform([features])
            proba = self.ml_model.predict_proba(features_scaled)[0, 1]
            score = proba
            
            if score >= self.optimal_threshold:
                classification = "Image GÉNÉRÉE PAR IA"
                confidence = (score - self.optimal_threshold) / (1 - self.optimal_threshold) * 100
            else:
                classification = "Image AUTHENTIQUE (Réelle)"
                confidence = (self.optimal_threshold - score) / self.optimal_threshold * 100
        else:
            score = self._heuristic_score(features)
            if score > 0.65:
                classification = "Image TRÈS PROBABLEMENT générée par IA"
                confidence = (score - 0.65) / 0.35 * 100
            elif score > 0.45:
                classification = "Image POSSIBLEMENT générée par IA"
                confidence = 50
            else:
                classification = "Image PROBABLEMENT authentique"
                confidence = (0.45 - score) / 0.45 * 100
        
        if verbose:
            print("\n" + "="*60)
            print("DÉTECTION D'IMAGE GÉNÉRÉE PAR IA")
            print("="*60)
            print(f"\nImage: {Path(image_path).name}")
            print(f"\nScore: {score:.3f} ({score*100:.1f}%)")
            print(f"Seuil de décision: {self.optimal_threshold:.3f}")
            print(f"Confiance: {confidence:.1f}%")
            print(f"\n>>> CLASSIFICATION: {classification}")
            print("="*60 + "\n")
        
        return score, classification
    
    def _heuristic_score(self, features):
        """Score heuristique de fallback"""
        score = (
            features[0] * 0.3 + features[5] * 0.2 + features[10] * 0.15 +
            features[15] * 0.2 + features[20] * 0.15
        )
        return np.clip(score, 0, 1)
    
    def save_model(self, path):
        """Sauvegarde du modèle"""
        model_data = {
            'ml_model': self.ml_model,
            'scaler': self.scaler,
            'optimal_threshold': self.optimal_threshold,
            'weights': self.weights,
            'is_trained': self.is_trained
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path):
        """Chargement du modèle"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.ml_model = model_data['ml_model']
        self.scaler = model_data['scaler']
        self.optimal_threshold = model_data['optimal_threshold']
        self.weights = model_data['weights']
        self.is_trained = model_data['is_trained']
        
        print(f"✓ Modèle chargé: {path}")
        print(f"  Seuil optimal: {self.optimal_threshold:.3f}")


# FONCTION WORKER POUR MULTIPROCESSING
def extract_features_worker(image_path):
    """
    Worker function pour extraction de features (doit être au niveau module pour multiprocessing)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_norm = gray.astype(np.float32) / 255.0
        h, w = gray_norm.shape
        
        # FFT
        fft = fft2(gray_norm)
        fft_shifted = fftshift(fft)
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        magnitude_log = np.log(magnitude + 1e-10)
        
        # Extraction simplifiée des features (même logique que dans la classe)
        features = []
        
        # HF features (5)
        center_h, center_w = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        radius = min(center_h, center_w) // 3
        mask_high = ((X - center_w)**2 + (Y - center_h)**2) > radius**2
        mag_hf = magnitude_log * mask_high
        phase_hf = phase * mask_high
        mag_norm = (mag_hf - np.mean(mag_hf)) / (np.std(mag_hf) + 1e-10)
        
        features.extend([
            np.sum(np.abs(mag_norm) > 2.5) / (h * w),
            np.std(mag_hf),
            np.mean(np.abs(mag_hf)),
            np.sum(np.abs(np.diff(phase_hf, axis=0)) > np.pi/2) / (h * w),
            np.max(mag_hf) - np.min(mag_hf)
        ])
        
        # PSD features (5)
        psd = magnitude ** 2
        r = np.sqrt((X - center_w)**2 + (Y - center_h)**2)
        r_int = r.astype(int)
        max_radius = min(int(np.min([center_h, center_w])), 100)
        radial_profile = np.zeros(max_radius)
        for i in range(max_radius):
            mask = (r_int == i)
            if np.sum(mask) > 0:
                radial_profile[i] = np.mean(psd[mask])
        
        if len(radial_profile) > 10:
            gradient = np.gradient(radial_profile)
            features.extend([
                np.var(gradient),
                np.sum(np.abs(np.diff(gradient)) > np.std(gradient) * 2) / len(gradient),
                np.mean(radial_profile[:10]),
                np.mean(radial_profile[-10:]),
                np.sum(psd) / (h * w)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, np.sum(psd) / (h * w)])
        
        # Upsampling features (5)
        freqs = [h//8, h//4, h//2]
        energies = []
        for freq in freqs:
            if freq < h and freq < w:
                window = 5
                fh = min(freq, h-window)
                fw = min(freq, w-window)
                local_energy = np.sum(magnitude[fh:fh+window, fw:fw+window] ** 2)
                energies.append(local_energy)
            else:
                energies.append(0)
        total_energy = np.sum(magnitude ** 2) + 1e-10
        features.extend([
            energies[0] / total_energy,
            energies[1] / total_energy,
            energies[2] / total_energy,
            sum(energies) / total_energy,
            np.std(energies)
        ])
        
        # Freq conv features (5)
        f_am = np.abs(fft_shifted)
        f_ph = np.angle(fft_shifted)
        kernel = np.ones((5, 5)) / 25
        f_am_log = np.log(f_am + 1e-10)
        f_am_conv = signal.convolve2d(f_am_log, kernel, mode='same', boundary='wrap')
        f_ph_conv = signal.convolve2d(f_ph, kernel, mode='same', boundary='wrap')
        features.extend([
            np.std(f_am_conv) / (np.mean(np.abs(f_am_conv)) + 1e-10),
            np.std(f_ph_conv) / (np.pi + 1e-10),
            np.var(f_am_conv),
            np.var(f_ph_conv),
            np.corrcoef(f_am_conv.flatten(), f_ph_conv.flatten())[0,1]
        ])
        
        # Phase features (5)
        phase_grad_h = np.angle(np.exp(1j * np.diff(phase, axis=0)))
        phase_grad_w = np.angle(np.exp(1j * np.diff(phase, axis=1)))
        features.extend([
            np.std(phase_grad_h),
            np.std(phase_grad_w),
            np.mean(np.abs(phase_grad_h)),
            np.mean(np.abs(phase_grad_w)),
            np.var(phase)
        ])
        
        # Global stats (5)
        features.extend([
            np.mean(magnitude_log),
            np.std(magnitude_log),
            np.mean(phase),
            np.std(phase),
            np.median(magnitude_log)
        ])
        
        # Multiscale features (5)
        scales = [1.0, 0.5, 0.25]
        energies = []
        for scale in scales:
            if scale < 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                img_scaled = cv2.resize(gray_norm, (new_w, new_h))
            else:
                img_scaled = gray_norm
            fft_scale = np.abs(fft2(img_scaled))
            energy = np.sum(fft_scale ** 2)
            energies.append(energy)
        features.extend([
            energies[0],
            energies[1] / (energies[0] + 1e-10),
            energies[2] / (energies[0] + 1e-10),
            np.std(energies),
            np.max(energies) / (np.min(energies) + 1e-10)
        ])
        
        return np.array(features)
        
    except Exception as e:
        return None


def setup_project_structure():
    """Crée la structure de dossiers recommandée"""
    structure = {
        'dataset': {
            'real_images': 'Images réelles/authentiques',
            'ai_generated': 'Images générées par IA',
        },
        'models': 'Modèles entraînés sauvegardés',
        'results': 'Résultats d\'analyses et rapports',
        'test_images': 'Images à tester individuellement'
    }
    
    print("\n" + "="*70)
    print("CRÉATION DE LA STRUCTURE DE PROJET")
    print("="*70 + "\n")
    
    for folder, subfolders in structure.items():
        if isinstance(subfolders, dict):
            Path(folder).mkdir(exist_ok=True)
            print(f"📁 {folder}/")
            for subfolder, desc in subfolders.items():
                path = Path(folder) / subfolder
                path.mkdir(exist_ok=True)
                print(f"   └── 📁 {subfolder}/ ({desc})")
        else:
            Path(folder).mkdir(exist_ok=True)
            print(f"📁 {folder}/ ({subfolders})")
    
    print("\n✓ Structure créée avec succès!")
    print("\nINSTRUCTIONS:")
    print("  1. Placez vos 1000 images RÉELLES dans: dataset/real_images/")
    print("  2. Placez vos 1000 images IA dans: dataset/ai_generated/")
    print("  3. Lancez l'entraînement avec: python script.py train")
    print("="*70 + "\n")


def train_detector(real_folder="dataset/real_images", 
                   fake_folder="dataset/ai_generated", 
                   model_name="ai_detector_model.pkl",
                   n_workers=None):
    """Entraîne le détecteur avec multiprocessing"""
    detector = AdaptiveAIDetector()
    
    real_path = Path(real_folder)
    fake_path = Path(fake_folder)
    
    if not real_path.exists() or not fake_path.exists():
        print("❌ ERREUR: Dossiers introuvables!")
        print(f"  Vérifiez que ces dossiers existent:")
        print(f"    - {real_folder}")
        print(f"    - {fake_folder}")
        return None
    
    # Vérification du contenu
    real_count = len(list(real_path.glob('*.jpg'))) + len(list(real_path.glob('*.png')))
    fake_count = len(list(fake_path.glob('*.jpg'))) + len(list(fake_path.glob('*.png')))
    
    if real_count == 0 or fake_count == 0:
        print("❌ ERREUR: Dossiers vides!")
        print(f"  Images réelles trouvées: {real_count}")
        print(f"  Images IA trouvées: {fake_count}")
        return None
    
    print(f"\n✓ Images trouvées:")
    print(f"  - Réelles: {real_count}")
    print(f"  - IA: {fake_count}")
    
    model_path = Path("models") / model_name
    Path("models").mkdir(exist_ok=True)
    
    results = detector.train(real_folder, fake_folder, test_size=0.2, 
                            save_model_path=str(model_path), n_workers=n_workers)
    
    # Sauvegarde du rapport
    report_path = Path("results") / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    Path("results").mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RAPPORT D'ENTRAÎNEMENT - DÉTECTEUR D'IMAGES IA\n")
        f.write("="*70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Images réelles: {real_count}\n")
        f.write(f"Images IA: {fake_count}\n")
        f.write(f"Workers parallèles: {n_workers if n_workers else cpu_count()-1}\n")
        f.write(f"Seuil optimal: {results['threshold']:.3f}\n")
        f.write(f"AUC-ROC: {results['auc']:.3f}\n")
        f.write(f"\nModèle sauvegardé: {model_path}\n")
    
    print(f"\n✓ Rapport sauvegardé: {report_path}")
    
    return detector


def predict_image(image_path, model_path="models/ai_detector_model.pkl"):
    """Prédiction sur une seule image"""
    model_file = Path(model_path)
    
    if not model_file.exists():
        print(f"❌ ERREUR: Modèle introuvable: {model_path}")
        print("   Entraînez d'abord le modèle avec: python script.py train")
        return None
    
    detector = AdaptiveAIDetector(model_path=model_path)
    score, classification = detector.predict(image_path, verbose=True)
    
    return score, classification


# SCRIPT PRINCIPAL
if __name__ == "__main__":
    import sys
    
    print("""
╔════════════════════════════════════════════════════════════════════╗
║    DÉTECTEUR D'IMAGES GÉNÉRÉES PAR IA - VERSION 2.0 TURBO ⚡       ║
║            Analyse Fréquentielle avec Multiprocessing              ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  1. Créer la structure:  python script.py setup")
        print("  2. Entraîner:           python script.py train [--workers N]")
        print("  3. Tester une image:    python script.py predict image.jpg")
        print("  4. Aide:                python script.py help")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "setup":
        setup_project_structure()
    
    elif command == "train":
        real_folder = "dataset/real_images"
        ai_folder = "dataset/ai_generated"
        n_workers = None
        
        # Parse arguments
        for i, arg in enumerate(sys.argv[2:]):
            if arg == "--workers" and i+3 < len(sys.argv):
                n_workers = int(sys.argv[i+3])
            elif not arg.startswith("--") and i == 0:
                real_folder = arg
            elif not arg.startswith("--") and i == 1:
                ai_folder = arg
        
        train_detector(real_folder, ai_folder, n_workers=n_workers)
    
    elif command == "predict":
        if len(sys.argv) < 3:
            print("❌ Spécifiez une image: python script.py predict image.jpg")
            sys.exit(1)
        
        image_path = sys.argv[2]
        model_path = "models/ai_detector_model.pkl"
        
        if len(sys.argv) > 3:
            model_path = sys.argv[3]
        
        result = predict_image(image_path, model_path)
        
        if result:
            score, classification = result
            print(f"\n{'='*60}")
            print(f"RÉSULTAT FINAL: {classification}")
            print(f"Score IA: {score*100:.1f}%")
            print(f"{'='*60}\n")
    
    elif command == "help":
        print(f"""
OPTIMISATIONS MULTIPROCESSING:
-------------------------------
Le script utilise maintenant ProcessPoolExecutor pour paralléliser l'extraction
de features sur tous vos CPU. Accélération typique: 4-8x plus rapide! ⚡

COMMANDES:
----------
  setup                              Créer la structure de dossiers
  train [real_dir] [ai_dir]          Entraîner avec tous les CPU
  train --workers 4                  Entraîner avec 4 workers spécifiques
  predict <image> [model]            Analyser une image
  help                               Afficher cette aide

EXEMPLES:
---------
  # Utiliser tous les CPU disponibles (recommandé)
  python script.py train

  # Limiter à 4 processus parallèles
  python script.py train --workers 4

  # Dossiers personnalisés avec 6 workers
  python script.py train ./mes_vraies_images ./mes_images_ia --workers 6

PERFORMANCE:
------------
  CPU: {cpu_count()} cores détectés
  Gain estimé: {cpu_count()-1}x plus rapide avec multiprocessing
  100 images: ~10-30 secondes (vs 3-10 minutes en séquentiel)
  2000 images: ~5-10 minutes (vs 1-2 heures en séquentiel)
        """)
    
    else:
        print(f"❌ Commande inconnue: {command}")
        print("   Utilisez: python script.py help")
