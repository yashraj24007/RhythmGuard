#!/usr/bin/env python3
"""
🫀 RhythmGuard CNN Quick Demo
============================
Quick demonstration script for CNN-based ECG rhythm classification.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.cnn_model import RhythmGuardCNN
from src.preprocessing.cnn_ecg_preprocessor import CNNECGPreprocessor
from src.models.severity_predictor import SeverityPredictor
from src.preprocessing.cnn_ecg_augmentor import ECGAugmentor

class RhythmGuardCNNDemo:
    """CNN Demo Class for RhythmGuard"""
    
    def __init__(self, data_path=".", model_path=None):
        """
        Initialize demo
        
        Args:
            data_path (str): Path to dataset
            model_path (str): Path to trained model (None to look for models)
        """
        self.data_path = Path(data_path)
        self.model_path = model_path
        
        # Initialize components
        self.cnn_model = None
        self.preprocessor = CNNECGPreprocessor(data_path)
        self.severity_predictor = SeverityPredictor()
        self.augmentor = ECGAugmentor()
        
        # Find model if not specified
        if self.model_path is None:
            self.model_path = self._find_model()
    
    def _find_model(self):
        """Find available CNN models"""
        model_files = list(Path('.').glob('*cnn*.h5'))
        
        if model_files:
            print(f"🔍 Found CNN models: {[m.name for m in model_files]}")
            return str(model_files[0])  # Use first found
        
        return None
    
    def load_model(self):
        """Load CNN model"""
        if self.model_path is None:
            print("❌ No CNN model found")
            print("💡 Train a model first using: python train_cnn.py")
            return False
        
        try:
            print(f"📁 Loading CNN model: {self.model_path}")
            
            # Initialize and load model
            self.cnn_model = RhythmGuardCNN()
            self.cnn_model.load_model(self.model_path)
            
            print(f"✅ Model loaded successfully")
            print(f"🎯 Classes: {self.cnn_model.class_names}")
            print(f"🔢 Parameters: {self.cnn_model.model.count_params():,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def demo_single_prediction(self, image_path=None):
        """Demonstrate single image prediction"""
        if self.cnn_model is None:
            print("❌ Model not loaded")
            return
        
        if image_path is None:
            # Find a random sample image
            image_path = self._get_random_sample()
        
        if image_path is None:
            print("❌ No sample images found")
            return
        
        print(f"\n🔍 Single Image Prediction Demo")
        print("=" * 40)
        print(f"📷 Image: {Path(image_path).name}")
        
        try:
            # Get prediction
            result = self.cnn_model.predict_single_image(image_path)
            
            # Get severity assessment
            severity_result = self.severity_predictor.predict_severity_rule_based(
                result['predicted_class']
            )
            
            # Display results
            print(f"\n🎯 Prediction Results:")
            print(f"   Class: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"   Severity: {severity_result['severity']}")
            print(f"   Description: {severity_result.get('description', '')}")
            
            # Show top probabilities
            print(f"\n📊 All Class Probabilities:")
            sorted_probs = sorted(result['all_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)
            
            for class_name, prob in sorted_probs:
                class_info = self.preprocessor.class_mapping.get(class_name, {})
                print(f"   {class_name} ({class_info.get('name', 'Unknown')}): {prob:.1%}")
            
            # Visualize image and prediction
            self._visualize_prediction(image_path, result, severity_result)
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
    
    def demo_batch_prediction(self, num_samples=5):
        """Demonstrate batch prediction on multiple images"""
        if self.cnn_model is None:
            print("❌ Model not loaded")
            return
        
        print(f"\n🔍 Batch Prediction Demo ({num_samples} samples)")
        print("=" * 50)
        
        # Get random samples from different classes
        samples = self._get_random_samples(num_samples)
        
        if not samples:
            print("❌ No sample images found")
            return
        
        results = []
        
        for i, (image_path, true_class) in enumerate(samples):
            try:
                print(f"\n📷 Sample {i+1}: {Path(image_path).name}")
                print(f"   True class: {true_class}")
                
                # Get prediction
                result = self.cnn_model.predict_single_image(image_path)
                
                # Get severity
                severity_result = self.severity_predictor.predict_severity_rule_based(
                    result['predicted_class']
                )
                
                # Check if correct
                is_correct = result['predicted_class'] == true_class
                
                print(f"   Predicted: {result['predicted_class']} ({result['confidence']:.1%})")
                print(f"   Severity: {severity_result['severity']}")
                print(f"   {'✅ Correct' if is_correct else '❌ Incorrect'}")
                
                results.append({
                    'image_path': image_path,
                    'true_class': true_class,
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence'],
                    'correct': is_correct,
                    'severity': severity_result['severity']
                })
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Summary
        if results:
            accuracy = sum(1 for r in results if r['correct']) / len(results)
            avg_confidence = np.mean([r['confidence'] for r in results])
            
            print(f"\n📊 Batch Summary:")
            print(f"   🎯 Accuracy: {accuracy:.1%}")
            print(f"   📈 Average confidence: {avg_confidence:.1%}")
            print(f"   ✅ Correct predictions: {sum(1 for r in results if r['correct'])}/{len(results)}")
    
    def demo_data_augmentation(self, num_samples=1):
        """Demonstrate data augmentation capabilities"""
        print(f"\n🎨 Data Augmentation Demo")
        print("=" * 30)
        
        # Get a sample image
        image_path = self._get_random_sample()
        
        if image_path is None:
            print("❌ No sample images found")
            return
        
        print(f"📷 Sample image: {Path(image_path).name}")
        
        # Show augmentation examples
        self.augmentor.visualize_augmentations(
            image_path,
            pipeline_type='moderate',
            num_augmentations=8,
            save_path='demo_augmentations.png'
        )
        
        # Compare different pipelines
        self.augmentor.compare_pipelines(
            image_path,
            save_path='demo_pipeline_comparison.png'
        )
    
    def demo_dataset_analysis(self):
        """Demonstrate dataset analysis capabilities"""
        print(f"\n📊 Dataset Analysis Demo")
        print("=" * 30)
        
        # Analyze training data
        if (self.data_path / 'train').exists():
            print(f"🔍 Analyzing training dataset...")
            train_stats = self.preprocessor.analyze_dataset('train')
            
            # Visualize samples
            self.preprocessor.visualize_samples(
                'train', 
                samples_per_class=3,
                save_path='demo_train_samples.png'
            )
        
        # Analyze test data
        if (self.data_path / 'test').exists():
            print(f"🔍 Analyzing test dataset...")
            test_stats = self.preprocessor.analyze_dataset('test')
    
    def demo_model_architecture(self):
        """Demonstrate model architecture visualization"""
        if self.cnn_model is None:
            print("❌ Model not loaded")
            return
        
        print(f"\n🏗️ Model Architecture Demo")
        print("=" * 30)
        
        # Print model summary
        print(f"📋 Model Summary:")
        self.cnn_model.model.summary()
        
        # Try to visualize model (if graphviz available)
        try:
            tf.keras.utils.plot_model(
                self.cnn_model.model,
                to_file='demo_model_architecture.png',
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=False,
                dpi=96
            )
            print(f"📊 Model architecture saved: demo_model_architecture.png")
        except Exception as e:
            print(f"⚠️ Could not visualize model architecture: {e}")
            print(f"💡 Install graphviz and pydot for model visualization")
    
    def _get_random_sample(self):
        """Get a random sample image"""
        for split in ['test', 'train']:
            split_path = self.data_path / split
            if split_path.exists():
                for class_folder in split_path.iterdir():
                    if class_folder.is_dir():
                        images = list(class_folder.glob('*.png'))
                        if images:
                            return str(random.choice(images))
        return None
    
    def _get_random_samples(self, num_samples):
        """Get multiple random samples from different classes"""
        samples = []
        
        for split in ['test', 'train']:
            split_path = self.data_path / split
            if not split_path.exists():
                continue
            
            for class_folder in split_path.iterdir():
                if class_folder.is_dir() and len(samples) < num_samples:
                    images = list(class_folder.glob('*.png'))
                    if images:
                        image_path = str(random.choice(images))
                        true_class = class_folder.name
                        samples.append((image_path, true_class))
            
            if len(samples) >= num_samples:
                break
        
        return samples[:num_samples]
    
    def _visualize_prediction(self, image_path, result, severity_result):
        """Visualize prediction results"""
        try:
            # Load and display image
            import cv2
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Show image
                ax1.imshow(image)
                ax1.set_title(f'ECG Image: {Path(image_path).name}')
                ax1.axis('off')
                
                # Show prediction bars
                classes = list(result['all_probabilities'].keys())
                probs = list(result['all_probabilities'].values())
                
                bars = ax2.barh(classes, probs)
                ax2.set_xlabel('Probability')
                ax2.set_title('Class Predictions')
                ax2.set_xlim(0, 1)
                
                # Highlight predicted class
                predicted_idx = classes.index(result['predicted_class'])
                bars[predicted_idx].set_color('red')
                
                plt.tight_layout()
                plt.savefig('demo_prediction.png', dpi=300, bbox_inches='tight')
                plt.show()
                
        except Exception as e:
            print(f"⚠️ Could not visualize prediction: {e}")
    
    def run_full_demo(self):
        """Run complete demonstration"""
        print("🫀 RhythmGuard CNN Complete Demo")
        print("=" * 60)
        
        # Load model
        if not self.load_model():
            return
        
        # Dataset analysis
        self.demo_dataset_analysis()
        
        # Model architecture
        self.demo_model_architecture()
        
        # Single prediction
        self.demo_single_prediction()
        
        # Batch prediction
        self.demo_batch_prediction(num_samples=5)
        
        # Data augmentation
        self.demo_data_augmentation()
        
        print(f"\n🎉 Complete demo finished!")
        print(f"📁 Demo outputs saved as demo_*.png files")


def main():
    """Main demo function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RhythmGuard CNN Demo')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained CNN model')
    parser.add_argument('--data_path', type=str, default='.',
                       help='Path to dataset directory')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'single', 'batch', 'augment', 'dataset', 'architecture'],
                       help='Demo mode to run')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = RhythmGuardCNNDemo(
        data_path=args.data_path,
        model_path=args.model_path
    )
    
    # Run selected demo mode
    if args.mode == 'full':
        demo.run_full_demo()
    elif args.mode == 'single':
        if demo.load_model():
            demo.demo_single_prediction()
    elif args.mode == 'batch':
        if demo.load_model():
            demo.demo_batch_prediction()
    elif args.mode == 'augment':
        demo.demo_data_augmentation()
    elif args.mode == 'dataset':
        demo.demo_dataset_analysis()
    elif args.mode == 'architecture':
        if demo.load_model():
            demo.demo_model_architecture()


if __name__ == "__main__":
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    main()