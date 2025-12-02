"""
Advanced Feature Extraction for AI Image Detection
==================================================

Extraction avancée de caractéristiques pour la détection d'images IA:
- Features spectrales FFT optimisées
- Features de texture (LBP, GLCM, Haralick)
- Features de couleur multi-espaces
- Features de bruit et compression
- Features géométriques et morphologiques

Author: Cholulaa
"""

import numpy as np
import cv2
from scipy import signal, ndimage
from scipy.fft import fft2, fftshift
from scipy.stats import entropy, skew, kurtosis
from skimage import feature, measure, filters, color
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureExtractor:
    """Extracteur de features avancé pour la détection IA"""
    
    def __init__(self, config: dict):
        self.config = config
        self.feature_names = []
        self._initialize_feature_names()
    
    def _initialize_feature_names(self):
        """Initialisation des noms de features pour l'interprétabilité"""
        categories = [
            'spectral_fft', 'spectral_dct', 'texture_lbp', 'texture_glcm',
            'color_histogram', 'color_moments', 'noise_analysis', 
            'edge_detection', 'morphological', 'compression_artifacts'
        ]
        
        for category in categories:
            if category == 'spectral_fft':
                self.feature_names.extend([
                    f'fft_high_freq_{i}' for i in range(8)
                ])
            elif category == 'spectral_dct':
                self.feature_names.extend([
                    f'dct_coeff_{i}' for i in range(6)
                ])
            # ... autres catégories
    
    def extract_all_features(self, image_path: str) -> np.ndarray:
        """Extraction complète de toutes les features"""
        try:
            # Chargement et validation de l'image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")
            
            # Préparation des canaux
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_norm = gray.astype(np.float32) / 255.0
            
            # Redimensionnement si nécessaire
            max_size = self.config.get('max_image_size', 1024)
            h, w = gray.shape
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                gray_norm = cv2.resize(gray_norm, (new_w, new_h))
                img = cv2.resize(img, (new_w, new_h))
            
            # Extraction par catégorie
            features = []
            
            # 1. Features spectrales FFT avancées (20 features)
            fft_features = self._extract_enhanced_fft_features(gray_norm)
            features.extend(fft_features)
            
            # 2. Features spectrales DCT (12 features)
            dct_features = self._extract_dct_features(gray_norm)
            features.extend(dct_features)
            
            # 3. Features de texture LBP (10 features)
            if self.config.get('use_texture_features', True):
                lbp_features = self._extract_lbp_features(gray)
                features.extend(lbp_features)
            
            # 4. Features de texture GLCM (8 features)
            if self.config.get('use_texture_features', True):
                glcm_features = self._extract_glcm_features(gray)
                features.extend(glcm_features)
            
            # 5. Features de couleur (12 features)
            if self.config.get('use_color_features', True):
                color_features = self._extract_color_features(img)
                features.extend(color_features)
            
            # 6. Features de bruit et compression (8 features)
            noise_features = self._extract_noise_features(gray_norm)
            features.extend(noise_features)
            
            # 7. Features de détection de contours (6 features)
            edge_features = self._extract_edge_features(gray)
            features.extend(edge_features)
            
            # 8. Features morphologiques (4 features)
            morph_features = self._extract_morphological_features(gray)
            features.extend(morph_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'extraction de features: {e}")
    
    def _extract_enhanced_fft_features(self, gray_norm: np.ndarray) -> List[float]:
        """Features spectrales FFT avancées"""
        h, w = gray_norm.shape
        
        # FFT 2D
        fft = fft2(gray_norm)
        fft_shifted = fftshift(fft)
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        magnitude_log = np.log(magnitude + 1e-10)
        
        features = []
        
        # Analyse par bandes de fréquence
        center_h, center_w = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        
        # Masques pour différentes bandes
        radius_low = min(center_h, center_w) // 8
        radius_mid = min(center_h, center_w) // 4
        radius_high = min(center_h, center_w) // 2
        
        dist = np.sqrt((X - center_w)**2 + (Y - center_h)**2)
        
        mask_low = dist <= radius_low
        mask_mid = (dist > radius_low) & (dist <= radius_mid)
        mask_high = (dist > radius_mid) & (dist <= radius_high)
        mask_very_high = dist > radius_high
        
        # Features par bande de fréquence
        for i, mask in enumerate([mask_low, mask_mid, mask_high, mask_very_high]):
            if np.sum(mask) > 0:
                mag_band = magnitude_log[mask]
                phase_band = phase[mask]
                
                features.extend([
                    np.mean(mag_band),           # Énergie moyenne
                    np.std(mag_band),            # Variabilité
                    np.max(mag_band) - np.min(mag_band),  # Dynamique
                    entropy(np.abs(mag_band) + 1e-10),    # Entropie
                    np.std(np.diff(phase_band))  # Cohérence de phase
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return features
    
    def _extract_dct_features(self, gray_norm: np.ndarray) -> List[float]:
        """Features basées sur la Transformée en Cosinus Discrète"""
        # DCT 2D
        dct = cv2.dct(gray_norm)
        
        features = []
        h, w = dct.shape
        
        # Analyse des coefficients DC et AC
        dc_coeff = dct[0, 0]
        features.append(dc_coeff)
        
        # Coefficients AC en zigzag (premiers coefficients)
        ac_coeffs = []
        for i in range(1, min(6, h)):
            for j in range(min(6-i, w)):
                if i + j < 6:
                    ac_coeffs.append(dct[i, j])
        
        ac_coeffs = np.array(ac_coeffs)
        
        features.extend([
            np.mean(ac_coeffs),
            np.std(ac_coeffs),
            np.sum(np.abs(ac_coeffs)),
            np.max(np.abs(ac_coeffs)),
            entropy(np.abs(ac_coeffs) + 1e-10)
        ])
        
        # Analyse par blocs 8x8 (comme JPEG)
        block_stats = []
        for i in range(0, h-7, 8):
            for j in range(0, w-7, 8):
                block = dct[i:i+8, j:j+8]
                block_energy = np.sum(block**2)
                block_stats.append(block_energy)
        
        if block_stats:
            block_stats = np.array(block_stats)
            features.extend([
                np.mean(block_stats),
                np.std(block_stats),
                np.max(block_stats) / (np.mean(block_stats) + 1e-10)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Coefficients haute fréquence (indicateurs de compression)
        hf_region = dct[h//2:, w//2:]
        features.extend([
            np.mean(np.abs(hf_region)),
            np.std(hf_region)
        ])
        
        return features