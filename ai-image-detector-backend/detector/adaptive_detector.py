# detector/adaptive_detector.py
import numpy as np
import cv2
from scipy import signal
from scipy.fft import fft2, fftshift
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report
import pickle
from pathlib import Path

class AdaptiveAIDetector:
    def __init__(self, model_path=None):
        self.weights = {'spectral_artifacts': 0.40, 'frequency_convolution': 0.35, 'phase_anomaly': 0.25}
        self.ml_model = None
        self.scaler = StandardScaler()
        self.optimal_threshold = 0.5
        self.is_trained = False
        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def extract_all_features(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Impossible de charger: {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_norm = gray.astype(np.float32) / 255.0
        h, w = gray_norm.shape
        fft = fft2(gray_norm)
        fft_shifted = fftshift(fft)
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        magnitude_log = np.log(magnitude + 1e-10)
        # (features extraction comme expliqué plus haut)
        # INSERTION ICI : récupère tout le code d’extraction de features du worker
        from detector.workers import extract_features_worker
        return extract_features_worker(image_path)

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.ml_model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10, min_samples_leaf=4, random_state=42, n_jobs=-1, class_weight='balanced')
        self.ml_model.fit(X_train_scaled, y_train)
        y_proba = self.ml_model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        self.optimal_threshold = thresholds[np.argmax(tpr - fpr)]
        self.is_trained = True
        y_pred = (y_proba >= self.optimal_threshold).astype(int)
        report = classification_report(y_test, y_pred, target_names=['Réelle', 'IA'], output_dict=True)
        return report, auc(fpr, tpr)

    def predict(self, image_path):
        features = self.extract_all_features(image_path)
        features_scaled = self.scaler.transform([features])
        proba = self.ml_model.predict_proba(features_scaled)[0, 1]
        if proba >= self.optimal_threshold:
            label = "IA"
        else:
            label = "RÉELLE"
        return proba, label

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'ml_model': self.ml_model,
                'scaler': self.scaler,
                'optimal_threshold': self.optimal_threshold,
                'weights': self.weights,
                'is_trained': self.is_trained
            }, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            mdl = pickle.load(f)
            self.ml_model = mdl['ml_model']
            self.scaler = mdl['scaler']
            self.optimal_threshold = mdl['optimal_threshold']
            self.weights = mdl['weights']
            self.is_trained = mdl['is_trained']
