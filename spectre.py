import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_magnitude_spectrum(image_path):
    """
    Calcule et affiche le spectre de magnitude d'une image
    
    Args:
        image_path: Chemin vers l'image à analyser
    """
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")
    
    # Appliquer la transformée de Fourier rapide (FFT)
    fft = np.fft.fft2(image)
    
    # Décaler le composant de fréquence zéro au centre
    fft_shift = np.fft.fftshift(fft)
    
    # Calculer le spectre de magnitude avec échelle logarithmique
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1)
    
    # Afficher les résultats
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Image originale
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Image Originale')
    axes[0].axis('off')
    
    # Spectre de magnitude
    axes[1].imshow(magnitude_spectrum, cmap='gray')
    axes[1].set_title('Spectre de Magnitude')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return magnitude_spectrum, fft_shift

# Utilisation
if __name__ == "__main__":
    # Remplacer par le chemin de votre image
    image_path = "test2.jpg"
    
    try:
        magnitude, fft_result = compute_magnitude_spectrum(image_path)
        print(f"Dimensions du spectre: {magnitude.shape}")
        print(f"Valeur min: {magnitude.min():.2f}, max: {magnitude.max():.2f}")
    except Exception as e:
        print(f"Erreur: {e}")
