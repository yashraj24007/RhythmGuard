#!/usr/bin/env python3
"""
🫀 RhythmGuard CNN Model Testing
===============================
Comprehensive testing script for CNN-based ECG rhythm classification.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path
import random

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cnn_model import RhythmGuardCNN
from cnn_ecg_preprocessor import CNNECGPreprocessor
from severity_predictor import SeverityPredictor

class CNNModelTester:
    """CNN Model Testing and Evaluation Class"""
    
    def __init__(self, model_path, data_path=".", target_size=(224, 224)):
        """
        Initialize CNN Model Tester
        
        Args:
            model_path (str): Path to trained CNN model
            data_path (str): Path to dataset
            target_size (tuple): Image target size
        """
        self.model_path = model_path
        self.data_path = Path(data_path)
        self.target_size = target_size
        
        # Initialize components
        self.cnn_model = None
        self.preprocessor = CNNECGPreprocessor(data_path, target_size)
        self.severity_predictor = SeverityPredictor()
        
        # Test results storage
        self.test_results = []
        self.predictions = []
        self.true_labels = []
        
    def load_model(self):
        """Load the trained CNN model"""
        print(f"📁 Loading CNN model from {self.model_path}")
        
        try:
            # Load the Keras model
            if self.model_path.endswith('.h5'):
                model = keras.models.load_model(self.model_path)
                
                # Try to load metadata
                metadata_path = self.model_path.replace('.h5', '_metadata.joblib')
                if os.path.exists(metadata_path):
                    metadata = joblib.load(metadata_path)
                    class_names = metadata.get('class_names', ['F', 'M', 'N', 'Q', 'S', 'V'])
                else:
                    class_names = ['F', 'M', 'N', 'Q', 'S', 'V']  # Default
                
                # Initialize RhythmGuardCNN with loaded model
                self.cnn_model = RhythmGuardCNN(
                    input_shape=model.input_shape[1:],
                    num_classes=model.output_shape[-1]
                )
                self.cnn_model.model = model
                self.cnn_model.class_names = class_names
                
            else:
                # Load using RhythmGuardCNN class
                self.cnn_model = RhythmGuardCNN()
                self.cnn_model.load_model(self.model_path)
            
            print(f"✅ Model loaded successfully")
            print(f"🎯 Classes: {self.cnn_model.class_names}")
            print(f"📊 Input shape: {self.cnn_model.model.input_shape}")
            print(f"🔢 Parameters: {self.cnn_model.model.count_params():,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def test_single_image(self, image_path, show_prediction=True):
        """Test a single ECG image"""
        if self.cnn_model is None or self.cnn_model.model is None:
            print("❌ Model not loaded")
            return None
        
        try:
            # Get prediction using CNN model
            result = self.cnn_model.predict_single_image(image_path)
            
            # Add severity prediction
            severity_result = self.severity_predictor.predict_severity_rule_based(
                result['predicted_class']
            )
            
            # Combine results
            combined_result = {
                'image_path': image_path,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'all_probabilities': result['all_probabilities'],
                'severity': severity_result['severity'],
                'severity_confidence': severity_result['confidence'],
                'severity_description': severity_result.get('description', '')
            }
            
            if show_prediction:
                print(f"📷 Image: {Path(image_path).name}")
                print(f"🎯 Predicted: {result['predicted_class']} ({result['confidence']:.1%})")
                print(f"🏥 Severity: {severity_result['severity']} ({severity_result['confidence']:.1%})")
                print(f"📝 Description: {severity_result.get('description', '')}")
            
            return combined_result
            
        except Exception as e:
            print(f"❌ Error testing image {image_path}: {e}")
            return None
    
    def test_dataset(self, split='test', max_samples_per_class=None, show_progress=True):
        """Test the model on a dataset split"""
        if self.cnn_model is None:
            print("❌ Model not loaded")
            return None
        
        print(f"🧪 Testing CNN model on {split} dataset")
        print("=" * 50)
        
        # Load test dataset
        try:
            X_test, y_test, class_names, image_paths = self.preprocessor.create_cnn_dataset(
                split=split,
                max_samples_per_class=max_samples_per_class,
                apply_augmentation=False
            )
            
            if len(X_test) == 0:
                print(f"❌ No test data found in {split} split")
                return None
            
            print(f"📊 Test dataset: {len(X_test)} samples, {len(class_names)} classes")
            
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            return None
        
        # Make predictions
        print(f"🔍 Making predictions...")
        try:
            predictions = self.cnn_model.model.predict(X_test, verbose=1)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Store results
            self.predictions = predictions
            self.true_labels = y_test
            
            # Calculate accuracy
            accuracy = np.mean(predicted_classes == y_test)
            
            print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy:.1%})")
            
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
            return None
        
        # Detailed analysis
        results = {
            'accuracy': accuracy,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_classes': y_test,
            'class_names': class_names,
            'num_samples': len(X_test)
        }
        
        return results
    
    def generate_classification_report(self, results):
        """Generate detailed classification report"""
        if results is None:
            return
        
        print(f"\n📈 Detailed Classification Report")
        print("=" * 50)
        
        # Classification report
        report = classification_report(
            results['true_classes'],
            results['predicted_classes'],
            target_names=results['class_names'],
            output_dict=True
        )
        
        print(classification_report(
            results['true_classes'],
            results['predicted_classes'],
            target_names=results['class_names']
        ))
        
        # Per-class analysis
        print(f"\n📊 Per-Class Performance:")
        for i, class_name in enumerate(results['class_names']):
            class_mask = results['true_classes'] == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(results['predicted_classes'][class_mask] == i)
                class_info = self.preprocessor.class_mapping.get(class_name, {})
                severity = class_info.get('severity', 'Unknown')
                
                print(f"  {class_name} ({class_info.get('name', 'Unknown')}):")
                print(f"    📊 Accuracy: {class_acc:.3f} ({class_acc:.1%})")
                print(f"    📁 Samples: {np.sum(class_mask)}")
                print(f"    🏥 Severity: {severity}")
        
        return report
    
    def plot_confusion_matrix(self, results, save_path=None):
        """Plot confusion matrix"""
        if results is None:
            return
        
        # Calculate confusion matrix
        cm = confusion_matrix(results['true_classes'], results['predicted_classes'])
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=results['class_names'],
            yticklabels=results['class_names']
        )
        plt.title('CNN Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved: {save_path}")
        
        plt.show()
        
        return cm
    
    def analyze_misclassifications(self, results, top_k=5):
        """Analyze most common misclassifications"""
        if results is None:
            return
        
        print(f"\n🔍 Misclassification Analysis")
        print("=" * 40)
        
        # Find misclassified samples
        misclassified = results['predicted_classes'] != results['true_classes']
        
        if np.sum(misclassified) == 0:
            print("🎉 No misclassifications found!")
            return
        
        print(f"❌ Total misclassifications: {np.sum(misclassified)} / {len(results['true_classes'])}")
        print(f"📊 Misclassification rate: {np.mean(misclassified):.1%}")
        
        # Analyze confusion pairs
        confusion_pairs = {}
        for true_idx, pred_idx in zip(results['true_classes'][misclassified], 
                                    results['predicted_classes'][misclassified]):
            pair = (true_idx, pred_idx)
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        # Sort by frequency
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🔝 Top {min(top_k, len(sorted_pairs))} Confusion Pairs:")
        for i, ((true_idx, pred_idx), count) in enumerate(sorted_pairs[:top_k]):
            true_class = results['class_names'][true_idx]
            pred_class = results['class_names'][pred_idx]
            
            true_info = self.preprocessor.class_mapping.get(true_class, {})
            pred_info = self.preprocessor.class_mapping.get(pred_class, {})
            
            print(f"  {i+1}. {true_class} → {pred_class}: {count} cases")
            print(f"     True: {true_info.get('name', 'Unknown')} ({true_info.get('severity', 'Unknown')})")
            print(f"     Pred: {pred_info.get('name', 'Unknown')} ({pred_info.get('severity', 'Unknown')})")
    
    def benchmark_performance(self, results):
        """Benchmark model performance"""
        if results is None:
            return
        
        print(f"\n⚡ Performance Benchmark")
        print("=" * 30)
        
        # Model size
        model_size = self.cnn_model.model.count_params()
        print(f"🔢 Model parameters: {model_size:,}")
        
        # Memory usage estimation
        input_size = np.prod(self.cnn_model.model.input_shape[1:])
        memory_mb = (model_size * 4 + input_size * 4) / (1024 * 1024)  # Rough estimate
        print(f"💾 Estimated memory: {memory_mb:.1f} MB")
        
        # Inference speed test
        print(f"⏱️ Testing inference speed...")
        test_input = np.random.random((1, *self.cnn_model.model.input_shape[1:]))
        
        import time
        times = []
        for _ in range(10):
            start = time.time()
            self.cnn_model.model.predict(test_input, verbose=0)
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        fps = 1000 / avg_time
        
        print(f"🚀 Average inference time: {avg_time:.1f} ms")
        print(f"📈 Throughput: {fps:.1f} FPS")
        
        # Clinical performance metrics
        print(f"\n🏥 Clinical Performance:")
        
        # Calculate sensitivity/specificity for critical conditions
        critical_classes = ['V', 'M']  # Ventricular and MI
        
        for class_name in critical_classes:
            if class_name in results['class_names']:
                class_idx = results['class_names'].index(class_name)
                
                # Binary classification metrics for this class
                true_positive = np.sum((results['true_classes'] == class_idx) & 
                                     (results['predicted_classes'] == class_idx))
                false_positive = np.sum((results['true_classes'] != class_idx) & 
                                      (results['predicted_classes'] == class_idx))
                false_negative = np.sum((results['true_classes'] == class_idx) & 
                                      (results['predicted_classes'] != class_idx))
                true_negative = np.sum((results['true_classes'] != class_idx) & 
                                     (results['predicted_classes'] != class_idx))
                
                sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
                
                class_info = self.preprocessor.class_mapping.get(class_name, {})
                print(f"  {class_name} ({class_info.get('name', 'Unknown')}):")
                print(f"    🎯 Sensitivity: {sensitivity:.3f} ({sensitivity:.1%})")
                print(f"    🛡️ Specificity: {specificity:.3f} ({specificity:.1%})")


def test_cnn_model(model_path, data_path=".", test_split='test', 
                  max_samples_per_class=None, save_results=True):
    """
    Main function to test CNN model
    
    Args:
        model_path (str): Path to trained CNN model
        data_path (str): Path to dataset
        test_split (str): Test dataset split name
        max_samples_per_class (int): Maximum samples per class for testing
        save_results (bool): Whether to save test results
    """
    
    print("🫀 RhythmGuard CNN Model Testing")
    print("=" * 60)
    
    # Initialize tester
    tester = CNNModelTester(model_path, data_path)
    
    # Load model
    if not tester.load_model():
        return False
    
    # Test on dataset
    results = tester.test_dataset(
        split=test_split,
        max_samples_per_class=max_samples_per_class
    )
    
    if results is None:
        return False
    
    # Generate reports
    tester.generate_classification_report(results)
    
    # Plot confusion matrix
    cm_path = f"cnn_confusion_matrix_{test_split}.png" if save_results else None
    tester.plot_confusion_matrix(results, cm_path)
    
    # Analyze misclassifications
    tester.analyze_misclassifications(results)
    
    # Benchmark performance
    tester.benchmark_performance(results)
    
    # Save results
    if save_results:
        results_path = f"cnn_test_results_{test_split}.joblib"
        joblib.dump(results, results_path)
        print(f"💾 Test results saved: {results_path}")
    
    print(f"\n🎉 CNN Model Testing Completed!")
    accuracy = results['accuracy']
    print(f"✅ Final Test Accuracy: {accuracy:.4f} ({accuracy:.1%})")
    
    return True


def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test RhythmGuard CNN Model')
    parser.add_argument('model_path', type=str, help='Path to trained CNN model')
    parser.add_argument('--data_path', type=str, default='.', 
                       help='Path to dataset directory')
    parser.add_argument('--test_split', type=str, default='test',
                       help='Test dataset split name')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples per class for testing')
    parser.add_argument('--no_save', action='store_true',
                       help='Don\'t save test results')
    
    args = parser.parse_args()
    
    # Test model
    success = test_cnn_model(
        model_path=args.model_path,
        data_path=args.data_path,
        test_split=args.test_split,
        max_samples_per_class=args.max_samples,
        save_results=not args.no_save
    )
    
    if success:
        print(f"\n✅ Testing completed successfully!")
    else:
        print(f"\n❌ Testing failed!")
        sys.exit(1)


if __name__ == "__main__":
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # For direct execution without arguments
    if len(sys.argv) == 1:
        # Look for trained models
        model_files = [f for f in os.listdir('.') if f.endswith('.h5') and 'cnn' in f.lower()]
        
        if model_files:
            print(f"🔍 Found CNN models: {model_files}")
            model_path = model_files[0]  # Use first found model
            print(f"📁 Using model: {model_path}")
            
            test_cnn_model(model_path)
        else:
            print("❌ No CNN model files found (looking for *.h5 files)")
            print("💡 Train a model first using: python train_cnn.py")
    else:
        main()