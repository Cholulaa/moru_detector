# detector/workers.py
import cv2
import numpy as np
from scipy.fft import fft2, fftshift
from scipy import signal

def extract_features_worker(image_path):
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_norm = gray.astype(np.float32) / 255.0
        h, w = gray_norm.shape
        fft = fft2(gray_norm)
        fft_shifted = fftshift(fft)
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        magnitude_log = np.log(magnitude + 1e-10)
        features = []
        # High Frequency (5)
        center_h, center_w = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        radius = min(center_h, center_w) // 3
        mask_high = ((X - center_w) ** 2 + (Y - center_h) ** 2) > radius ** 2
        mag_hf = magnitude_log * mask_high
        phase_hf = phase * mask_high
        mag_norm = (mag_hf - np.mean(mag_hf)) / (np.std(mag_hf) + 1e-10)
        features.extend([
            np.sum(np.abs(mag_norm) > 2.5) / (h * w),
            np.std(mag_hf),
            np.mean(np.abs(mag_hf)),
            np.sum(np.abs(np.diff(phase_hf, axis=0)) > np.pi / 2) / (h * w),
            np.max(mag_hf) - np.min(mag_hf)
        ])
        # PSD features (5)
        psd = magnitude ** 2
        r = np.sqrt((X - center_w) ** 2 + (Y - center_h) ** 2)
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
        freqs = [h // 8, h // 4, h // 2]
        energies = []
        for freq in freqs:
            if freq < h and freq < w:
                window = 5
                fh = min(freq, h - window)
                fw = min(freq, w - window)
                local_energy = np.sum(magnitude[fh:fh + window, fw:fw + window] ** 2)
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
        # Frequency convolution features (5)
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
            np.corrcoef(f_am_conv.flatten(), f_ph_conv.flatten())[0, 1]
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
        # Global statistics (5)
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
    except Exception:
        return None
