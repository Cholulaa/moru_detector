import numpy as np
import cv2
from scipy import signal
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import warnings
warnings.filterwarnings('ignore')

class AIGeneratedImageDetector:
    """
    Détecteur hybride d'images générées par IA basé sur l'analyse de Fourier
    Combine trois approches: analyse spectrale, convolutions fréquentielles, et détection d'artefacts
    """
    
    def __init__(self):
        self.weights = {
            'spectral_artifacts': 0.40,
            'frequency_convolution': 0.35,
            'phase_anomaly': 0.25
        }
        
    def preprocess_image(self, image_path):
        """
        Prétraitement de l'image pour l'analyse
        
        Args:
            image_path: Chemin vers l'image
        Returns:
            Image en niveaux de gris normalisée
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        # Conversion en niveaux de gris et normalisation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_normalized = gray.astype(np.float32) / 255.0
        
        return gray_normalized, img
    
    def compute_fft_components(self, image):
        """
        Calcule les composantes FFT (magnitude et phase)
        
        Args:
            image: Image normalisée
        Returns:
            Tuple (magnitude_spectrum, phase_spectrum, fft_shifted)
        """
        # Transformée de Fourier 2D
        fft = fft2(image)
        fft_shifted = fftshift(fft)
        
        # Extraction magnitude et phase
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        
        # Échelle logarithmique pour magnitude
        magnitude_log = np.log(magnitude + 1e-10)
        
        return magnitude_log, phase, fft_shifted
    
    def extract_high_frequency_features(self, magnitude, phase, image_shape):
        """
        APPROCHE 1: Extraction des caractéristiques haute fréquence
        Basée sur FreqNet - détection d'artefacts dans les HF
        
        Args:
            magnitude: Spectre de magnitude
            phase: Spectre de phase
            image_shape: Dimensions de l'image
        Returns:
            Score d'artefacts haute fréquence (0-1)
        """
        h, w = image_shape
        center_h, center_w = h // 2, w // 2
        
        # Création d'un masque passe-haut (périphérie du spectre)
        Y, X = np.ogrid[:h, :w]
        radius = min(center_h, center_w) // 3
        mask_low = ((X - center_w)**2 + (Y - center_h)**2) <= radius**2
        mask_high = ~mask_low
        
        # Extraction des composantes haute fréquence
        magnitude_hf = magnitude * mask_high
        phase_hf = phase * mask_high
        
        # Détection de patterns anormaux (grilles, rayures)
        # Les GAN produisent des artefacts géométriques réguliers
        magnitude_hf_normalized = (magnitude_hf - np.mean(magnitude_hf)) / (np.std(magnitude_hf) + 1e-10)
        
        # Détection de pics anormaux
        threshold_peaks = 2.5
        anomalous_peaks = np.sum(np.abs(magnitude_hf_normalized) > threshold_peaks)
        peak_density = anomalous_peaks / (h * w)
        
        # Analyse de la régularité spatiale (artefacts de grille)
        # Les images synthétiques présentent souvent des motifs périodiques
        fft_of_magnitude = np.abs(fft2(magnitude_hf))
        fft_of_magnitude_normalized = (fft_of_magnitude - np.mean(fft_of_magnitude)) / (np.std(fft_of_magnitude) + 1e-10)
        
        periodicity_score = np.sum(np.abs(fft_of_magnitude_normalized) > 3.0) / fft_of_magnitude.size
        
        # Détection d'anomalies de phase
        phase_variance = np.var(phase_hf[mask_high])
        phase_discontinuities = np.sum(np.abs(np.diff(phase_hf, axis=0)) > np.pi/2) + \
                               np.sum(np.abs(np.diff(phase_hf, axis=1)) > np.pi/2)
        phase_anomaly_score = phase_discontinuities / (h * w * 2)
        
        # Score combiné d'artefacts
        artifact_score = np.clip(
            0.4 * peak_density * 1000 + 
            0.3 * periodicity_score * 500 + 
            0.3 * phase_anomaly_score * 100,
            0, 1
        )
        
        return artifact_score
    
    def frequency_domain_convolution(self, fft_shifted, image_shape):
        """
        APPROCHE 3: Convolutions dans le domaine fréquentiel
        Inspirée de FreqNet - apprentissage spectral
        
        Args:
            fft_shifted: FFT décalée de l'image
            image_shape: Dimensions de l'image
        Returns:
            Score de convolution fréquentielle (0-1)
        """
        # Séparation amplitude et phase
        f_am = np.abs(fft_shifted)
        f_ph = np.angle(fft_shifted)
        
        # Filtre convolutionnel simple dans le domaine fréquentiel
        # Simule les couches convolutionnelles de FreqNet
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        # Convolution sur l'amplitude
        f_am_log = np.log(f_am + 1e-10)
        f_am_conv = signal.convolve2d(f_am_log, kernel, mode='same', boundary='wrap')
        
        # Convolution sur la phase
        f_ph_conv = signal.convolve2d(f_ph, kernel, mode='same', boundary='wrap')
        
        # Reconstruction et analyse
        f_reconstructed = np.exp(f_am_conv) * np.exp(1j * f_ph_conv)
        
        # iFFT pour retourner au domaine spatial
        img_reconstructed = np.abs(ifft2(ifftshift(f_reconstructed)))
        
        # Analyse de la différence spectrale
        # Les images réelles ont une distribution spectrale plus uniforme
        am_variance = np.var(f_am_conv)
        ph_variance = np.var(f_ph_conv)
        
        # Détection d'irrégularités après convolution
        # Images synthétiques montrent des patterns résiduels spécifiques
        spectral_irregularity = np.std(f_am_conv) / (np.mean(np.abs(f_am_conv)) + 1e-10)
        phase_irregularity = np.std(f_ph_conv) / (np.pi + 1e-10)
        
        # Score de convolution fréquentielle
        freq_conv_score = np.clip(
            0.5 * min(spectral_irregularity / 2.0, 1.0) + 
            0.5 * min(phase_irregularity / 0.5, 1.0),
            0, 1
        )
        
        return freq_conv_score
    
    def analyze_power_spectral_density(self, magnitude, image_shape):
        """
        Analyse de la densité spectrale de puissance (PSD)
        Détection d'anomalies dans la distribution d'énergie
        
        Args:
            magnitude: Spectre de magnitude
            image_shape: Dimensions de l'image
        Returns:
            Score PSD (0-1)
        """
        h, w = image_shape
        
        # Calcul de la PSD
        psd = magnitude ** 2
        
        # Analyse radiale de la PSD
        center_h, center_w = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        r = np.sqrt((X - center_w)**2 + (Y - center_h)**2)
        
        # Binning radial
        r_int = r.astype(int)
        max_radius = int(np.min([center_h, center_w]))
        
        radial_profile = np.zeros(max_radius)
        for i in range(max_radius):
            mask = (r_int == i)
            if np.sum(mask) > 0:
                radial_profile[i] = np.mean(psd[mask])
        
        # Les images synthétiques ont souvent des chutes anormales dans la PSD
        # Détection de non-linéarités
        if len(radial_profile) > 10:
            radial_gradient = np.gradient(radial_profile)
            gradient_variance = np.var(radial_gradient)
            
            # Recherche de discontinuités
            discontinuities = np.sum(np.abs(np.diff(radial_gradient)) > np.std(radial_gradient) * 2)
            
            psd_anomaly = min(gradient_variance / 100.0, 1.0) * 0.6 + \
                         min(discontinuities / 20.0, 1.0) * 0.4
        else:
            psd_anomaly = 0.5
        
        return np.clip(psd_anomaly, 0, 1)
    
    def detect_upsampling_artifacts(self, magnitude, image_shape):
        """
        Détection spécifique d'artefacts d'upsampling des GANs
        Les opérations d'upsampling laissent des signatures fréquentielles
        
        Args:
            magnitude: Spectre de magnitude
            image_shape: Dimensions de l'image
        Returns:
            Score d'artefacts d'upsampling (0-1)
        """
        h, w = image_shape
        
        # Les artefacts d'upsampling apparaissent à des fréquences spécifiques
        # correspondant aux facteurs d'upsampling (2x, 4x, 8x)
        upsampling_frequencies = [h//8, h//4, h//2, w//8, w//4, w//2]
        
        artifact_energy = 0
        for freq in upsampling_frequencies:
            if freq < h and freq < w:
                # Extraction d'énergie autour des fréquences suspectes
                window_size = 3
                freq_h = min(freq, h-window_size)
                freq_w = min(freq, w-window_size)
                
                local_energy = np.sum(magnitude[
                    freq_h:freq_h+window_size,
                    freq_w:freq_w+window_size
                ] ** 2)
                artifact_energy += local_energy
        
        # Normalisation
        total_energy = np.sum(magnitude ** 2)
        upsampling_score = artifact_energy / (total_energy + 1e-10)
        
        return np.clip(upsampling_score * 5, 0, 1)
    
    def compute_phase_coherence(self, phase):
        """
        Analyse de la cohérence de phase
        Les images synthétiques ont souvent des phases moins cohérentes
        
        Args:
            phase: Spectre de phase
        Returns:
            Score d'incohérence de phase (0-1)
        """
        # Calcul des gradients de phase
        phase_grad_h = np.diff(phase, axis=0)
        phase_grad_w = np.diff(phase, axis=1)
        
        # Wrapping des gradients de phase [-π, π]
        phase_grad_h = np.angle(np.exp(1j * phase_grad_h))
        phase_grad_w = np.angle(np.exp(1j * phase_grad_w))
        
        # Cohérence = uniformité des gradients
        coherence_h = np.std(phase_grad_h)
        coherence_w = np.std(phase_grad_w)
        
        # Score d'incohérence (plus élevé = plus suspect)
        incoherence_score = (coherence_h + coherence_w) / (2 * np.pi)
        
        return np.clip(incoherence_score, 0, 1)
    
    def predict(self, image_path, verbose=True):
        """
        Prédiction principale: détermine si l'image est générée par IA
        
        Args:
            image_path: Chemin vers l'image
            verbose: Afficher les détails
        Returns:
            Tuple (score_final, scores_détaillés, classification)
        """
        # Prétraitement
        image_gray, image_color = self.preprocess_image(image_path)
        h, w = image_gray.shape
        
        # Calcul des composantes fréquentielles
        magnitude, phase, fft_shifted = self.compute_fft_components(image_gray)
        
        # APPROCHE 1: Analyse spectrale multi-domaines
        artifact_score = self.extract_high_frequency_features(magnitude, phase, (h, w))
        psd_score = self.analyze_power_spectral_density(magnitude, (h, w))
        upsampling_score = self.detect_upsampling_artifacts(magnitude, (h, w))
        
        spectral_score = (artifact_score * 0.5 + psd_score * 0.3 + upsampling_score * 0.2)
        
        # APPROCHE 3: Convolutions fréquentielles
        freq_conv_score = self.frequency_domain_convolution(fft_shifted, (h, w))
        
        # Analyse de cohérence de phase
        phase_coherence_score = self.compute_phase_coherence(phase)
        
        # Score final pondéré
        final_score = (
            self.weights['spectral_artifacts'] * spectral_score +
            self.weights['frequency_convolution'] * freq_conv_score +
            self.weights['phase_anomaly'] * phase_coherence_score
        )
        
        # Calibration finale (ajustement empirique)
        final_score = np.clip(final_score * 1.2, 0, 1)
        
        # Classification
        if final_score > 0.65:
            classification = "Image TRÈS PROBABLEMENT générée par IA"
        elif final_score > 0.45:
            classification = "Image POSSIBLEMENT générée par IA"
        else:
            classification = "Image PROBABLEMENT authentique"
        
        scores_detailed = {
            'spectral_artifacts': spectral_score,
            'frequency_convolution': freq_conv_score,
            'phase_anomaly': phase_coherence_score,
            'artifact_detection': artifact_score,
            'psd_analysis': psd_score,
            'upsampling_artifacts': upsampling_score
        }
        
        if verbose:
            print("\n" + "="*60)
            print("ANALYSE DE DÉTECTION D'IMAGE GÉNÉRÉE PAR IA")
            print("="*60)
            print(f"\nImage: {image_path}")
            print(f"Dimensions: {w}x{h} pixels")
            print("\n--- Scores Détaillés ---")
            print(f"Artefacts spectraux:        {spectral_score:.3f}")
            print(f"  - Artefacts HF:          {artifact_score:.3f}")
            print(f"  - Analyse PSD:           {psd_score:.3f}")
            print(f"  - Artefacts upsampling:  {upsampling_score:.3f}")
            print(f"Convolution fréquentielle: {freq_conv_score:.3f}")
            print(f"Anomalie de phase:         {phase_coherence_score:.3f}")
            print("\n" + "-"*60)
            print(f"SCORE FINAL: {final_score:.3f} ({final_score*100:.1f}%)")
            print(f"CLASSIFICATION: {classification}")
            print("="*60 + "\n")
        
        return final_score, scores_detailed, classification


# Fonction d'utilisation simple
def analyze_image_for_ai(image_path):
    """
    Fonction simple pour analyser une image
    
    Args:
        image_path: Chemin vers l'image à analyser
    Returns:
        Score de probabilité que l'image soit générée par IA (0-1)
    """
    detector = AIGeneratedImageDetector()
    score, details, classification = detector.predict(image_path, verbose=True)
    return score


# Exemple d'utilisation
if __name__ == "__main__":
    # Remplacer par le chemin de votre image
    image_path = "image1.png"
    
    try:
        # Analyse de l'image
        ai_probability = analyze_image_for_ai(image_path)
        
        print(f"\n✓ Probabilité que l'image soit générée par IA: {ai_probability*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        import traceback
        traceback.print_exc()
