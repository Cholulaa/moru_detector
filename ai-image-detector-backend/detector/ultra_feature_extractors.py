"""
Ultra Advanced Feature Extractors for AI Image Detection
========================================================

Extracteurs de features ultra-avancés utilisant:
- Wavelets multi-échelles
- Filtres de Gabor optimisés
- Analyse morphologique avancée
- Features statistiques complexes
- Analyse de compression et artefacts

Author: Enhanced by RovoDev AI
License: MIT
"""

import numpy as np
import cv2
import pywt
from scipy import signal, ndimage, stats
from scipy.fft import fft2, fftshift, dct
from skimage import feature, measure, filters, morphology, segmentation
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')


class BaseFeatureExtractor:
    """Classe de base pour tous les extracteurs"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def extract(self, processed_images: Dict[str, np.ndarray]) -> List[float]:
        """Méthode abstraite à implémenter"""
        raise NotImplementedError


class SpectralFeatureExtractor(BaseFeatureExtractor):
    """Extracteur de features spectrales avancées"""
    
    def extract(self, processed_images: Dict[str, np.ndarray]) -> List[float]:
        features = []
        
        for name, img in processed_images.items():
            # FFT Analysis
            fft = fft2(img)
            fft_shifted = fftshift(fft)
            magnitude = np.abs(fft_shifted)
            phase = np.angle(fft_shifted)
            
            # Features spectrales de base
            features.extend(self._extract_fft_features(magnitude, phase))
            
            # DCT Analysis
            features.extend(self._extract_dct_features(img))
            
            # PSD Analysis
            features.extend(self._extract_psd_features(magnitude))
        
        return features
    
    def _extract_fft_features(self, magnitude: np.ndarray, phase: np.ndarray) -> List[float]:
        """Features FFT avancées"""
        features = []
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Analyse par bandes de fréquence
        Y, X = np.ogrid[:h, :w]
        radii = [min(center_h, center_w) // i for i in [8, 4, 2, 1.5]]
        
        for i, radius in enumerate(radii):
            mask = ((X - center_w) ** 2 + (Y - center_h) ** 2) <= radius ** 2
            mag_band = magnitude * mask
            phase_band = phase * mask
            
            if np.sum(mask) > 0:
                features.extend([
                    np.mean(mag_band[mask]),
                    np.std(mag_band[mask]),
                    np.sum(mag_band > np.percentile(mag_band[mask], 95)) / np.sum(mask),
                    np.std(phase_band[mask]),
                    stats.skew(mag_band[mask].flatten()),
                    stats.kurtosis(mag_band[mask].flatten())
                ])
            else:
                features.extend([0.0] * 6)
        
        # Features de régularité spectrale
        mag_log = np.log(magnitude + 1e-10)
        features.extend([
            np.mean(mag_log),
            np.std(mag_log),
            stats.entropy(magnitude.flatten()),
            np.corrcoef(magnitude.flatten(), phase.flatten())[0, 1] if magnitude.size > 1 else 0,
        ])
        
        return features
    
    def _extract_dct_features(self, img: np.ndarray) -> List[float]:
        """Features DCT (Discrete Cosine Transform)"""
        dct_coeffs = dct(dct(img, axis=0), axis=1)
        
        # Analyse des coefficients DCT
        features = []
        h, w = dct_coeffs.shape
        
        # Coefficients basse fréquence (coin supérieur gauche)
        low_freq = dct_coeffs[:h//4, :w//4]
        features.extend([
            np.mean(low_freq),
            np.std(low_freq),
            np.sum(np.abs(low_freq) > np.percentile(np.abs(low_freq), 90))
        ])
        
        # Coefficients haute fréquence
        high_freq = dct_coeffs[h//2:, w//2:]
        features.extend([
            np.mean(np.abs(high_freq)),
            np.std(high_freq),
            np.sum(np.abs(high_freq) > 1e-3) / high_freq.size
        ])
        
        return features
    
    def _extract_psd_features(self, magnitude: np.ndarray) -> List[float]:
        """Features Power Spectral Density"""
        psd = magnitude ** 2
        h, w = psd.shape
        center_h, center_w = h // 2, w // 2
        
        # Profil radial
        Y, X = np.ogrid[:h, :w]
        r = np.sqrt((X - center_w) ** 2 + (Y - center_h) ** 2)
        r_int = r.astype(int)
        max_radius = min(center_h, center_w, 100)
        
        radial_profile = np.zeros(max_radius)
        for i in range(max_radius):
            mask = (r_int == i)
            if np.sum(mask) > 0:
                radial_profile[i] = np.mean(psd[mask])
        
        # Features du profil radial
        if len(radial_profile) > 10:
            gradient = np.gradient(radial_profile)
            return [
                np.var(gradient),
                np.sum(np.abs(np.diff(gradient)) > np.std(gradient) * 2) / len(gradient),
                np.mean(radial_profile[:5]),
                np.mean(radial_profile[-5:]),
                stats.pearsonr(range(len(radial_profile)), radial_profile)[0] if len(radial_profile) > 2 else 0
            ]
        else:
            return [0.0] * 5


class WaveletFeatureExtractor(BaseFeatureExtractor):
    """Extracteur de features ondelettes multi-échelles"""
    
    def extract(self, processed_images: Dict[str, np.ndarray]) -> List[float]:
        features = []
        
        if not self.config['feature_extraction']['use_wavelets']:
            return []
        
        wavelet_types = self.config['feature_extraction']['wavelet_types']
        
        for name, img in processed_images.items():
            for wavelet in wavelet_types:
                try:
                    # Décomposition en ondelettes
                    coeffs = pywt.wavedec2(img, wavelet, level=4)
                    
                    # Analyse des coefficients
                    features.extend(self._analyze_wavelet_coeffs(coeffs))
                    
                except Exception:
                    # Si erreur avec ce type d'ondelette, utiliser des zeros
                    features.extend([0.0] * 15)
        
        return features
    
    def _analyze_wavelet_coeffs(self, coeffs) -> List[float]:
        """Analyse des coefficients d'ondelettes"""
        features = []
        
        # Coefficient d'approximation
        cA = coeffs[0]
        features.extend([
            np.mean(cA),
            np.std(cA),
            stats.skew(cA.flatten()),
            stats.kurtosis(cA.flatten())
        ])
        
        # Coefficients de détail
        for level_coeffs in coeffs[1:]:
            cH, cV, cD = level_coeffs
            
            # Energie des détails
            energy_h = np.sum(cH ** 2)
            energy_v = np.sum(cV ** 2)
            energy_d = np.sum(cD ** 2)
            total_energy = energy_h + energy_v + energy_d + 1e-10
            
            features.extend([
                energy_h / total_energy,
                energy_v / total_energy,
                energy_d / total_energy
            ])
            
            # Statistiques des coefficients de détail
            all_details = np.concatenate([cH.flatten(), cV.flatten(), cD.flatten()])
            features.extend([
                np.std(all_details),
                np.sum(np.abs(all_details) > np.percentile(np.abs(all_details), 95)) / len(all_details)
            ])
            
            break  # Ne prendre que le premier niveau pour éviter trop de features
        
        return features[:15]  # Limiter à 15 features par ondelette


class TextureFeatureExtractor(BaseFeatureExtractor):
    """Extracteur de features de texture avancées"""
    
    def extract(self, processed_images: Dict[str, np.ndarray]) -> List[float]:
        features = []
        
        for name, img in processed_images.items():
            # Conversion en uint8 pour certaines fonctions
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Local Binary Pattern
            features.extend(self._extract_lbp_features(img_uint8))
            
            # GLCM (Gray Level Co-occurrence Matrix)
            features.extend(self._extract_glcm_features(img_uint8))
            
            # Filtres de Gabor
            if self.config['feature_extraction']['use_gabor_filters']:
                features.extend(self._extract_gabor_features(img))
            
            break  # Une seule image pour éviter redondance
        
        return features
    
    def _extract_lbp_features(self, img: np.ndarray) -> List[float]:
        """Features Local Binary Pattern"""
        radius = 3
        n_points = 8 * radius
        
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        
        # Histogramme LBP
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-10)
        
        # Features statistiques du LBP
        return [
            np.mean(hist),
            np.std(hist),
            stats.skew(hist),
            stats.kurtosis(hist),
            np.sum(hist > np.mean(hist)) / len(hist)
        ]
    
    def _extract_glcm_features(self, img: np.ndarray) -> List[float]:
        """Features GLCM (Gray Level Co-occurrence Matrix)"""
        # Réduire les niveaux de gris pour accélérer GLCM
        img_reduced = (img // 32).astype(np.uint8)
        
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        features = []
        
        try:
            glcm = greycomatrix(img_reduced, distances, angles, levels=8, symmetric=True, normed=True)
            
            # Propriétés GLCM
            contrast = greycoprops(glcm, 'contrast')
            dissimilarity = greycoprops(glcm, 'dissimilarity')
            homogeneity = greycoprops(glcm, 'homogeneity')
            energy = greycoprops(glcm, 'energy')
            
            features.extend([
                np.mean(contrast),
                np.std(contrast),
                np.mean(dissimilarity),
                np.mean(homogeneity),
                np.mean(energy)
            ])
            
        except Exception:
            features.extend([0.0] * 5)
        
        return features
    
    def _extract_gabor_features(self, img: np.ndarray) -> List[float]:
        """Features filtres de Gabor"""
        features = []
        
        frequencies = self.config['feature_extraction']['gabor_frequencies']
        orientations = self.config['feature_extraction']['gabor_orientations']
        
        for freq in frequencies:
            for orientation in orientations:
                try:
                    # Application du filtre de Gabor
                    real, _ = filters.gabor(img, frequency=freq, theta=np.deg2rad(orientation))
                    
                    # Statistiques de la réponse
                    features.extend([
                        np.mean(np.abs(real)),
                        np.std(real),
                        np.sum(np.abs(real) > np.percentile(np.abs(real), 90)) / real.size
                    ])
                    
                except Exception:
                    features.extend([0.0] * 3)
                    
                # Limiter le nombre de features
                if len(features) >= 24:  # 8 combinaisons * 3 features
                    break
            if len(features) >= 24:
                break
        
        return features[:24]


class MorphologicalFeatureExtractor(BaseFeatureExtractor):
    """Extracteur de features morphologiques"""
    
    def extract(self, processed_images: Dict[str, np.ndarray]) -> List[float]:
        features = []
        
        img = processed_images['original']
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Features de forme et structure
        features.extend(self._extract_shape_features(img_uint8))
        features.extend(self._extract_edge_features(img))
        features.extend(self._extract_morphological_operations(img_uint8))
        
        return features
    
    def _extract_shape_features(self, img: np.ndarray) -> List[float]:
        """Features de forme et géométrie"""
        # Détection de contours
        edges = cv2.Canny(img, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return [0.0] * 8
        
        # Analyse des contours
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100]
        perimeters = [cv2.arcLength(c, True) for c in contours if cv2.contourArea(c) > 100]
        
        if len(areas) == 0:
            return [0.0] * 8
        
        return [
            np.mean(areas),
            np.std(areas),
            np.mean(perimeters),
            np.std(perimeters),
            len(contours),
            np.mean([a/p**2 if p > 0 else 0 for a, p in zip(areas, perimeters)]),  # Circularité
            np.sum(edges > 0) / edges.size,  # Densité de contours
            np.std([len(c) for c in contours])  # Complexité des contours
        ]
    
    def _extract_edge_features(self, img: np.ndarray) -> List[float]:
        """Features de détection de bords"""
        # Différents détecteurs de bords
        sobel_h = filters.sobel_h(img)
        sobel_v = filters.sobel_v(img)
        sobel_mag = np.sqrt(sobel_h**2 + sobel_v**2)
        
        laplacian = filters.laplacian(img)
        
        return [
            np.mean(np.abs(sobel_mag)),
            np.std(sobel_mag),
            np.sum(sobel_mag > np.percentile(sobel_mag, 95)) / sobel_mag.size,
            np.mean(np.abs(laplacian)),
            np.std(laplacian),
            stats.skew(sobel_mag.flatten()),
            np.corrcoef(sobel_h.flatten(), sobel_v.flatten())[0,1] if sobel_h.size > 1 else 0
        ]
    
    def _extract_morphological_operations(self, img: np.ndarray) -> List[float]:
        """Features d'opérations morphologiques"""
        kernel = np.ones((3,3), np.uint8)
        
        # Opérations morphologiques
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        
        # Différences avec l'image originale
        diff_open = np.mean(np.abs(img.astype(float) - opened.astype(float)))
        diff_close = np.mean(np.abs(img.astype(float) - closed.astype(float)))
        
        return [
            diff_open,
            diff_close,
            np.mean(gradient),
            np.std(gradient),
            np.sum(gradient > np.percentile(gradient, 90)) / gradient.size
        ]


class FrequencyFeatureExtractor(BaseFeatureExtractor):
    """Extracteur de features fréquentielles spécialisées"""
    
    def extract(self, processed_images: Dict[str, np.ndarray]) -> List[float]:
        features = []
        
        img = processed_images['original']
        
        # Features de périodicité
        features.extend(self._extract_periodicity_features(img))
        
        # Features de régularité fréquentielle
        features.extend(self._extract_regularity_features(img))
        
        return features
    
    def _extract_periodicity_features(self, img: np.ndarray) -> List[float]:
        """Détection de motifs périodiques"""
        fft = fft2(img)
        magnitude = np.abs(fftshift(fft))
        
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Analyse des pics de fréquence
        peaks = feature.peak_local_maxima(magnitude.flatten())[0]
        peak_values = magnitude.flatten()[peaks]
        
        if len(peak_values) > 1:
            peak_ratio = np.max(peak_values) / np.mean(peak_values)
            peak_std = np.std(peak_values)
        else:
            peak_ratio = 1.0
            peak_std = 0.0
        
        # Symétrie spectrale
        left_half = magnitude[:, :center_w]
        right_half = np.fliplr(magnitude[:, center_w:])
        
        min_width = min(left_half.shape[1], right_half.shape[1])
        symmetry = np.corrcoef(
            left_half[:, -min_width:].flatten(),
            right_half[:, :min_width].flatten()
        )[0,1] if min_width > 0 else 0
        
        return [
            len(peaks) / magnitude.size,
            peak_ratio,
            peak_std,
            symmetry,
            np.var(magnitude) / np.mean(magnitude)**2  # Coefficient de variation
        ]
    
    def _extract_regularity_features(self, img: np.ndarray) -> List[float]:
        """Features de régularité dans le domaine fréquentiel"""
        # Analyse multi-échelle
        features = []
        
        for scale in [1.0, 0.75, 0.5]:
            if scale < 1.0:
                h_new = int(img.shape[0] * scale)
                w_new = int(img.shape[1] * scale)
                img_scaled = cv2.resize(img, (w_new, h_new))
            else:
                img_scaled = img
            
            fft = fft2(img_scaled)
            magnitude = np.abs(fft)
            
            # Entropie spectrale
            magnitude_norm = magnitude / np.sum(magnitude)
            spectral_entropy = stats.entropy(magnitude_norm.flatten())
            
            # Concentration d'énergie
            energy_ratio = np.sum(magnitude > np.percentile(magnitude, 95)) / magnitude.size
            
            features.extend([spectral_entropy, energy_ratio])
        
        return features


class StatisticalFeatureExtractor(BaseFeatureExtractor):
    """Extracteur de features statistiques avancées"""
    
    def extract(self, processed_images: Dict[str, np.ndarray]) -> List[float]:
        features = []
        
        for name, img in processed_images.items():
            # Features statistiques de base
            features.extend(self._extract_basic_stats(img))
            
            # Features de distribution
            features.extend(self._extract_distribution_features(img))
            
            if name == 'original':  # Eviter la redondance
                break
        
        return features
    
    def _extract_basic_stats(self, img: np.ndarray) -> List[float]:
        """Statistiques de base améliorées"""
        pixels = img.flatten()
        
        return [
            np.mean(pixels),
            np.std(pixels),
            stats.skew(pixels),
            stats.kurtosis(pixels),
            np.median(pixels),
            stats.iqr(pixels),  # Interquartile range
            np.percentile(pixels, 25),
            np.percentile(pixels, 75),
            np.min(pixels),
            np.max(pixels)
        ]
    
    def _extract_distribution_features(self, img: np.ndarray) -> List[float]:
        """Features de distribution avancées"""
        pixels = img.flatten()
        
        # Histogramme
        hist, _ = np.histogram(pixels, bins=50, range=(0, 1))
        hist = hist.astype(float) / np.sum(hist)
        
        # Entropie de l'histogramme
        hist_entropy = stats.entropy(hist + 1e-10)
        
        # Test de normalité
        try:
            _, shapiro_p = stats.shapiro(pixels[:5000])  # Echantillon pour la vitesse
        except:
            shapiro_p = 1.0
        
        # Moments d'ordre supérieur
        moment_3 = stats.moment(pixels, moment=3)
        moment_4 = stats.moment(pixels, moment=4)
        
        return [
            hist_entropy,
            shapiro_p,
            moment_3,
            moment_4,
            np.var(hist),  # Variation de l'histogramme
            np.sum(hist > np.mean(hist)) / len(hist)  # Proportion de bins au-dessus de la moyenne
        ]