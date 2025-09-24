"""
RythmGuard Severity Prediction Module
===================================

This module provides functionality to predict severity levels (Mild, Moderate, Severe)
for detected arrhythmias in ECG images.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class SeverityPredictor:
    """
    ECG Arrhythmia Severity Prediction Class
    """
    
    def __init__(self):
        """Initialize Severity Predictor"""
        self.severity_mapping = {
            0: 'Mild',
            1: 'Moderate', 
            2: 'Severe'
        }
        
        self.class_severity_rules = {
            'N': {'mild': 0.8, 'moderate': 0.15, 'severe': 0.05},  # Normal - mostly mild
            'S': {'mild': 0.6, 'moderate': 0.3, 'severe': 0.1},   # Supraventricular - mixed
            'V': {'mild': 0.4, 'moderate': 0.4, 'severe': 0.2},   # Ventricular - more serious
            'F': {'mild': 0.3, 'moderate': 0.5, 'severe': 0.2},   # Fusion - moderate to severe
            'Q': {'mild': 0.5, 'moderate': 0.3, 'severe': 0.2},   # Unknown - uncertain
            'M': {'mild': 0.1, 'moderate': 0.3, 'severe': 0.6}    # MI - mostly severe
        }
        
        self.model = None
        self.feature_extractor = ECGFeatureExtractor()
    
    def extract_severity_features(self, image, ecg_class):
        """
        Extract features that indicate severity from ECG image
        
        Args:
            image (numpy.ndarray): ECG image
            ecg_class (str): ECG classification (N, S, V, F, Q, M)
            
        Returns:
            numpy.ndarray: Feature vector for severity prediction
        """
        # Extract basic image features
        basic_features = self.feature_extractor.extract_basic_features(image)
        
        # Extract ECG-specific features
        ecg_features = self.feature_extractor.extract_ecg_features(image, ecg_class)
        
        # Combine features
        features = np.concatenate([basic_features, ecg_features])
        
        return features
    
    def generate_severity_labels(self, ecg_classes, num_samples=None):
        """
        Generate severity labels based on ECG class distribution
        
        Args:
            ecg_classes (list): List of ECG classes
            num_samples (int): Number of samples (if None, uses length of ecg_classes)
            
        Returns:
            numpy.ndarray: Severity labels (0: Mild, 1: Moderate, 2: Severe)
        """
        if num_samples is None:
            num_samples = len(ecg_classes)
        
        severity_labels = []
        
        for ecg_class in ecg_classes:
            # Get probability distribution for this class
            probs = self.class_severity_rules.get(ecg_class, {'mild': 0.33, 'moderate': 0.33, 'severe': 0.34})
            
            # Sample severity based on probabilities
            severity = np.random.choice(['mild', 'moderate', 'severe'], 
                                      p=[probs['mild'], probs['moderate'], probs['severe']])
            
            # Convert to numeric label
            severity_numeric = {'mild': 0, 'moderate': 1, 'severe': 2}[severity]
            severity_labels.append(severity_numeric)
        
        return np.array(severity_labels)
    
    def train_severity_model(self, X_features, y_severity):
        """
        Train severity prediction model
        
        Args:
            X_features (numpy.ndarray): Feature matrix
            y_severity (numpy.ndarray): Severity labels
        """
        print("🔬 Training Severity Prediction Model...")
        
        # Use Random Forest for initial model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        # Train model
        self.model.fit(X_features, y_severity)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_features, y_severity, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
            print(f"Top 5 most important features: {np.argsort(feature_importance)[-5:]}")
    
    def predict_severity(self, image, ecg_class):
        """
        Predict severity for a single ECG image
        
        Args:
            image (numpy.ndarray): ECG image
            ecg_class (str): ECG classification
            
        Returns:
            tuple: (severity_label, confidence, severity_name)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_severity_model first.")
        
        # Extract features
        features = self.extract_severity_features(image, ecg_class)
        features = features.reshape(1, -1)
        
        # Predict
        severity_label = self.model.predict(features)[0]
        confidence = np.max(self.model.predict_proba(features))
        severity_name = self.severity_mapping[severity_label]
        
        return severity_label, confidence, severity_name
    
    def predict_severity_rule_based(self, ecg_class):
        """
        Predict severity using rule-based approach (no trained model required)
        
        Args:
            ecg_class (str): ECG classification (N, S, V, F, Q, M)
            
        Returns:
            dict: {'severity': severity_name, 'confidence': confidence}
        """
        if ecg_class not in self.class_severity_rules:
            return {'severity': 'Moderate', 'confidence': 0.5}
        
        # Get probabilities for this class
        probs = self.class_severity_rules[ecg_class]
        
        # Find most likely severity
        severity_probs = [probs['mild'], probs['moderate'], probs['severe']]
        severity_label = np.argmax(severity_probs)
        confidence = max(severity_probs)
        severity_name = self.severity_mapping[severity_label]
        
        return {
            'severity': severity_name,
            'confidence': confidence
        }
    
    def get_clinical_priority(self, severity_label):
        """
        Get clinical priority based on severity
        
        Args:
            severity_label (int): Severity label (0=Mild, 1=Moderate, 2=Severe)
            
        Returns:
            str: Clinical priority
        """
        priority_mapping = {
            0: 'Low Priority',     # Mild
            1: 'Medium Priority',  # Moderate  
            2: 'High Priority'     # Severe
        }
        return priority_mapping.get(severity_label, 'Medium Priority')
    
    def save_model(self, filepath):
        """Save trained model"""
        joblib.dump(self.model, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        print(f"Model loaded from: {filepath}")


class ECGFeatureExtractor:
    """
    ECG Feature Extraction Class
    """
    
    def extract_basic_features(self, image):
        """
        Extract basic image features
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Basic feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        features = []
        
        # Statistical features
        features.extend([
            np.mean(gray),           # Mean intensity
            np.std(gray),            # Standard deviation
            np.var(gray),            # Variance
            np.min(gray),            # Minimum intensity
            np.max(gray),            # Maximum intensity
            np.median(gray),         # Median intensity
        ])
        
        # Histogram features
        hist, _ = np.histogram(gray, bins=16, range=(0, 1))
        hist = hist / np.sum(hist)  # Normalize
        features.extend(hist.tolist())
        
        # Gradient features
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(grad_magnitude),
            np.std(grad_magnitude),
            np.max(grad_magnitude)
        ])
        
        return np.array(features)
    
    def extract_ecg_features(self, image, ecg_class):
        """
        Extract ECG-specific features
        
        Args:
            image (numpy.ndarray): ECG image
            ecg_class (str): ECG class
            
        Returns:
            numpy.ndarray: ECG-specific feature vector
        """
        features = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Signal amplitude features
        row_means = np.mean(gray, axis=1)
        col_means = np.mean(gray, axis=0)
        
        features.extend([
            np.std(row_means),       # Vertical variation
            np.std(col_means),       # Horizontal variation
            np.max(row_means) - np.min(row_means),  # Amplitude range
        ])
        
        # Frequency domain features (approximation)
        fft_row = np.fft.fft(row_means)
        fft_power = np.abs(fft_row)**2
        
        features.extend([
            np.sum(fft_power[:len(fft_power)//4]),   # Low frequency power
            np.sum(fft_power[len(fft_power)//4:len(fft_power)//2]),  # High frequency power
        ])
        
        # Class-specific features
        class_features = self._get_class_specific_features(gray, ecg_class)
        features.extend(class_features)
        
        return np.array(features)
    
    def _get_class_specific_features(self, image, ecg_class):
        """
        Get class-specific features
        
        Args:
            image (numpy.ndarray): Grayscale ECG image
            ecg_class (str): ECG class
            
        Returns:
            list: Class-specific features
        """
        features = []
        
        if ecg_class == 'V':  # Ventricular - wide QRS
            # Look for wider signal patterns
            features.append(self._measure_signal_width(image))
            features.append(self._measure_signal_irregularity(image))
        
        elif ecg_class == 'M':  # Myocardial Infarction - ST changes
            # Look for baseline shifts
            features.append(self._measure_baseline_shift(image))
            features.append(self._measure_signal_elevation(image))
        
        elif ecg_class == 'S':  # Supraventricular - rapid rate
            # Look for rapid patterns
            features.append(self._measure_signal_frequency(image))
            features.append(self._measure_rhythm_regularity(image))
        
        else:
            # Default features for other classes
            features.extend([0.0, 0.0])
        
        return features
    
    def _measure_signal_width(self, image):
        """Measure average signal width"""
        row_means = np.mean(image, axis=1)
        # Find signal peaks and measure width
        peaks = row_means > (np.mean(row_means) + 0.5 * np.std(row_means))
        if np.any(peaks):
            peak_widths = []
            in_peak = False
            width = 0
            for peak in peaks:
                if peak and not in_peak:
                    in_peak = True
                    width = 1
                elif peak and in_peak:
                    width += 1
                elif not peak and in_peak:
                    peak_widths.append(width)
                    in_peak = False
            return np.mean(peak_widths) if peak_widths else 0.0
        return 0.0
    
    def _measure_signal_irregularity(self, image):
        """Measure signal irregularity"""
        col_means = np.mean(image, axis=0)
        diff = np.diff(col_means)
        return np.std(diff)
    
    def _measure_baseline_shift(self, image):
        """Measure baseline shift"""
        row_means = np.mean(image, axis=1)
        baseline = np.mean(row_means)
        return np.abs(np.mean(row_means) - baseline)
    
    def _measure_signal_elevation(self, image):
        """Measure signal elevation"""
        row_means = np.mean(image, axis=1)
        return np.max(row_means) - np.mean(row_means)
    
    def _measure_signal_frequency(self, image):
        """Measure signal frequency"""
        col_means = np.mean(image, axis=0)
        zero_crossings = np.sum(np.diff(np.sign(col_means - np.mean(col_means))) != 0)
        return zero_crossings / len(col_means)
    
    def _measure_rhythm_regularity(self, image):
        """Measure rhythm regularity"""
        col_means = np.mean(image, axis=0)
        intervals = np.diff(np.where(col_means > np.mean(col_means) + np.std(col_means))[0])
        return np.std(intervals) if len(intervals) > 1 else 0.0