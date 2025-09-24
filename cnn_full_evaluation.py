#!/usr/bin/env python3
"""
🫀 RhythmGuard CNN Evaluation Suite
==================================
Comprehensive evaluation and testing suite for CNN-based ECG classification.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import joblib
from pathlib import Path
import time
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cnn_model import RhythmGuardCNN
from cnn_ecg_preprocessor import CNNECGPreprocessor
from severity_predictor import SeverityPredictor

class CNNModelEvaluator:
    """Comprehensive CNN Model Evaluation Class"""
    
    def __init__(self, model_path, data_path=".", output_dir="evaluation_results"):
        """
        Initialize CNN Model Evaluator
        
        Args:
            model_path (str): Path to trained CNN model
            data_path (str): Path to dataset
            output_dir (str): Directory to save evaluation results
        """
        self.model_path = model_path
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.cnn_model = None
        self.preprocessor = CNNECGPreprocessor(data_path)
        self.severity_predictor = SeverityPredictor()
        
        # Evaluation results storage
        self.evaluation_results = {}
    
    def load_model(self):
        """Load CNN model for evaluation"""
        print(f"📁 Loading CNN model: {self.model_path}")
        
        try:
            self.cnn_model = RhythmGuardCNN()
            self.cnn_model.load_model(self.model_path)
            
            print(f"✅ Model loaded successfully")
            print(f"🎯 Classes: {self.cnn_model.class_names}")
            print(f"🔢 Parameters: {self.cnn_model.model.count_params():,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def evaluate_test_set(self, test_split='test', save_predictions=True):
        """Evaluate model on test set"""
        if self.cnn_model is None:
            print("❌ Model not loaded")
            return None
        
        print(f"🧪 Evaluating on {test_split} dataset")
        print("=" * 40)
        
        try:
            # Load test data
            X_test, y_test, class_names, image_paths = self.preprocessor.create_cnn_dataset(
                split=test_split,
                apply_augmentation=False
            )
            
            if len(X_test) == 0:
                print(f"❌ No test data found")
                return None
            
            print(f"📊 Test samples: {len(X_test)}")
            print(f"🎯 Classes: {class_names}")
            
            # Make predictions
            print(f"🔍 Making predictions...")
            predictions = self.cnn_model.model.predict(X_test, verbose=1)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Calculate metrics
            accuracy = np.mean(predicted_classes == y_test)
            
            # Store results
            results = {
                'test_split': test_split,
                'accuracy': accuracy,
                'predictions': predictions,
                'predicted_classes': predicted_classes,
                'true_classes': y_test,
                'class_names': class_names,
                'image_paths': image_paths,
                'num_samples': len(X_test)
            }
            
            self.evaluation_results['test_evaluation'] = results
            
            print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy:.1%})")
            
            # Save predictions if requested
            if save_predictions:
                pred_df = pd.DataFrame({
                    'image_path': image_paths,
                    'true_class': [class_names[i] for i in y_test],
                    'predicted_class': [class_names[i] for i in predicted_classes],
                    'confidence': [predictions[i][predicted_classes[i]] for i in range(len(predictions))],
                    'correct': predicted_classes == y_test
                })
                
                pred_file = self.output_dir / f'predictions_{test_split}.csv'
                pred_df.to_csv(pred_file, index=False)
                print(f"💾 Predictions saved: {pred_file}")
            
            return results
            
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return None
    
    def generate_classification_report(self, results, save_report=True):
        """Generate detailed classification report"""
        if results is None:
            return None
        
        print(f"\n📈 Classification Report")
        print("=" * 40)
        
        # Generate sklearn classification report
        report_dict = classification_report(
            results['true_classes'],
            results['predicted_classes'],
            target_names=results['class_names'],
            output_dict=True
        )
        
        # Print report
        print(classification_report(
            results['true_classes'],
            results['predicted_classes'],
            target_names=results['class_names']
        ))
        
        # Create detailed DataFrame
        report_df = pd.DataFrame(report_dict).transpose()
        
        # Add clinical information
        clinical_info = []
        for class_name in results['class_names']:
            class_info = self.preprocessor.class_mapping.get(class_name, {})
            clinical_info.append({
                'class': class_name,
                'full_name': class_info.get('name', 'Unknown'),
                'severity': class_info.get('severity', 'Unknown'),
                'description': class_info.get('description', '')
            })
        
        clinical_df = pd.DataFrame(clinical_info)
        
        if save_report:
            # Save detailed report
            report_file = self.output_dir / 'classification_report.csv'
            report_df.to_csv(report_file)
            
            clinical_file = self.output_dir / 'clinical_info.csv'
            clinical_df.to_csv(clinical_file, index=False)
            
            print(f"💾 Classification report saved: {report_file}")
            print(f"💾 Clinical info saved: {clinical_file}")
        
        return report_dict, clinical_df
    
    def plot_confusion_matrix(self, results, normalize=False, save_plot=True):
        """Plot and analyze confusion matrix"""
        if results is None:
            return None
        
        print(f"\n📊 Confusion Matrix Analysis")
        print("=" * 30)
        
        # Calculate confusion matrix
        cm = confusion_matrix(results['true_classes'], results['predicted_classes'])
        
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_display = cm_norm
            fmt = '.2f'
            title_suffix = ' (Normalized)'
        else:
            cm_display = cm
            fmt = 'd'
            title_suffix = ''
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=results['class_names'],
            yticklabels=results['class_names'],
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        
        plt.title(f'CNN Confusion Matrix{title_suffix}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_plot:
            suffix = '_normalized' if normalize else ''
            cm_file = self.output_dir / f'confusion_matrix{suffix}.png'
            plt.savefig(cm_file, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved: {cm_file}")
        
        plt.show()
        
        # Analyze confusion matrix
        self._analyze_confusion_matrix(cm, results['class_names'])
        
        return cm
    
    def _analyze_confusion_matrix(self, cm, class_names):
        """Analyze confusion matrix for insights"""
        print(f"\n🔍 Confusion Matrix Insights:")
        
        # Most confused pairs
        confusion_pairs = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append((
                        class_names[i], 
                        class_names[j], 
                        cm[i, j],
                        cm[i, j] / cm[i].sum() * 100  # Percentage
                    ))
        
        # Sort by count
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"🔝 Top 5 Confusion Pairs:")
        for i, (true_class, pred_class, count, percentage) in enumerate(confusion_pairs[:5]):
            true_info = self.preprocessor.class_mapping.get(true_class, {})
            pred_info = self.preprocessor.class_mapping.get(pred_class, {})
            
            print(f"  {i+1}. {true_class} → {pred_class}: {count} cases ({percentage:.1f}%)")
            print(f"     {true_info.get('name', 'Unknown')} → {pred_info.get('name', 'Unknown')}")
    
    def plot_roc_curves(self, results, save_plot=True):
        """Plot ROC curves for multiclass classification"""
        if results is None:
            return None
        
        print(f"\n📈 ROC Curve Analysis")
        print("=" * 25)
        
        try:
            # Binarize labels for multiclass ROC
            y_test_bin = label_binarize(results['true_classes'], 
                                       classes=range(len(results['class_names'])))
            n_classes = len(results['class_names'])
            
            # Compute ROC curve and AUC for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], 
                                            results['predictions'][:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot ROC curves
            plt.figure(figsize=(12, 10))
            
            colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
            
            for i, color in zip(range(n_classes), colors):
                class_name = results['class_names'][i]
                class_info = self.preprocessor.class_mapping.get(class_name, {})
                
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{class_name} ({class_info.get("name", "Unknown")}) '
                              f'(AUC = {roc_auc[i]:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves - Multiclass Classification')
            plt.legend(loc="lower right", bbox_to_anchor=(1.3, 0))
            plt.tight_layout()
            
            if save_plot:
                roc_file = self.output_dir / 'roc_curves.png'
                plt.savefig(roc_file, dpi=300, bbox_inches='tight')
                print(f"📈 ROC curves saved: {roc_file}")
            
            plt.show()
            
            # Print AUC summary
            print(f"📊 AUC Scores:")
            for i, class_name in enumerate(results['class_names']):
                class_info = self.preprocessor.class_mapping.get(class_name, {})
                print(f"   {class_name} ({class_info.get('name', 'Unknown')}): {roc_auc[i]:.3f}")
            
            avg_auc = np.mean(list(roc_auc.values()))
            print(f"   📈 Average AUC: {avg_auc:.3f}")
            
            return roc_auc
            
        except Exception as e:
            print(f"❌ ROC curve error: {e}")
            return None
    
    def clinical_performance_analysis(self, results, save_analysis=True):
        """Analyze model performance from clinical perspective"""
        if results is None:
            return None
        
        print(f"\n🏥 Clinical Performance Analysis")
        print("=" * 40)
        
        # Analyze by severity levels
        severity_analysis = {}
        
        for class_name in results['class_names']:
            class_info = self.preprocessor.class_mapping.get(class_name, {})
            severity = class_info.get('severity', 'Unknown')
            
            if severity not in severity_analysis:
                severity_analysis[severity] = {
                    'classes': [],
                    'true_positive': 0,
                    'false_positive': 0,
                    'false_negative': 0,
                    'true_negative': 0
                }
            
            severity_analysis[severity]['classes'].append(class_name)
        
        # Calculate metrics by severity
        class_idx_map = {name: idx for idx, name in enumerate(results['class_names'])}
        
        for severity, info in severity_analysis.items():
            for class_name in info['classes']:
                class_idx = class_idx_map[class_name]
                
                # Binary classification metrics for this class
                tp = np.sum((results['true_classes'] == class_idx) & 
                           (results['predicted_classes'] == class_idx))
                fp = np.sum((results['true_classes'] != class_idx) & 
                           (results['predicted_classes'] == class_idx))
                fn = np.sum((results['true_classes'] == class_idx) & 
                           (results['predicted_classes'] != class_idx))
                tn = np.sum((results['true_classes'] != class_idx) & 
                           (results['predicted_classes'] != class_idx))
                
                info['true_positive'] += tp
                info['false_positive'] += fp
                info['false_negative'] += fn
                info['true_negative'] += tn
        
        # Calculate clinical metrics
        clinical_metrics = {}
        
        print(f"📊 Performance by Severity Level:")
        for severity, info in severity_analysis.items():
            tp, fp, fn, tn = info['true_positive'], info['false_positive'], info['false_negative'], info['true_negative']
            
            # Calculate metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            clinical_metrics[severity] = {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'f1_score': f1_score,
                'classes': info['classes']
            }
            
            print(f"\n  {severity} Severity:")
            print(f"    Classes: {', '.join(info['classes'])}")
            print(f"    🎯 Sensitivity (Recall): {sensitivity:.3f} ({sensitivity:.1%})")
            print(f"    🛡️ Specificity: {specificity:.3f} ({specificity:.1%})")
            print(f"    📊 Precision: {precision:.3f} ({precision:.1%})")
            print(f"    📈 F1-Score: {f1_score:.3f}")
        
        # Critical condition analysis
        critical_classes = ['V', 'M']  # Ventricular and MI
        critical_performance = {}
        
        print(f"\n🚨 Critical Condition Analysis:")
        for class_name in critical_classes:
            if class_name in results['class_names']:
                class_idx = class_idx_map[class_name]
                class_info = self.preprocessor.class_mapping.get(class_name, {})
                
                # Calculate metrics
                tp = np.sum((results['true_classes'] == class_idx) & 
                           (results['predicted_classes'] == class_idx))
                fp = np.sum((results['true_classes'] != class_idx) & 
                           (results['predicted_classes'] == class_idx))
                fn = np.sum((results['true_classes'] == class_idx) & 
                           (results['predicted_classes'] != class_idx))
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                
                critical_performance[class_name] = {
                    'sensitivity': sensitivity,
                    'precision': precision,
                    'missed_cases': fn,
                    'false_alarms': fp
                }
                
                print(f"\n  {class_name} ({class_info.get('name', 'Unknown')}):")
                print(f"    🎯 Sensitivity: {sensitivity:.3f} (Critical - should be high)")
                print(f"    📊 Precision: {precision:.3f}")
                print(f"    ❌ Missed cases: {fn}")
                print(f"    ⚠️ False alarms: {fp}")
                
                if sensitivity < 0.8:
                    print(f"    🚨 WARNING: Low sensitivity for critical condition!")
        
        if save_analysis:
            # Save clinical analysis
            clinical_file = self.output_dir / 'clinical_analysis.json'
            with open(clinical_file, 'w') as f:
                json.dump({
                    'severity_analysis': {k: {**v, 'classes': v['classes']} 
                                        for k, v in clinical_metrics.items()},
                    'critical_performance': critical_performance
                }, f, indent=2)
            
            print(f"💾 Clinical analysis saved: {clinical_file}")
        
        return clinical_metrics, critical_performance
    
    def performance_benchmark(self, results, num_trials=10):
        """Benchmark model performance (speed, memory)"""
        if self.cnn_model is None or results is None:
            return None
        
        print(f"\n⚡ Performance Benchmark")
        print("=" * 30)
        
        # Model size metrics
        model_size = self.cnn_model.model.count_params()
        model_memory = self.cnn_model.model.count_params() * 4 / (1024 * 1024)  # MB (float32)
        
        print(f"🔢 Model Parameters: {model_size:,}")
        print(f"💾 Model Memory: {model_memory:.1f} MB")
        
        # Inference speed test
        print(f"⏱️ Testing inference speed ({num_trials} trials)...")
        
        # Create test input
        test_input = np.random.random((1, *self.cnn_model.model.input_shape[1:]))
        
        # Warmup
        for _ in range(3):
            self.cnn_model.model.predict(test_input, verbose=0)
        
        # Timing trials
        times = []
        for _ in range(num_trials):
            start_time = time.time()
            self.cnn_model.model.predict(test_input, verbose=0)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000
        fps = 1000 / avg_time
        
        print(f"🚀 Inference Time:")
        print(f"   Average: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"   Range: {min_time:.2f} - {max_time:.2f} ms")
        print(f"   Throughput: {fps:.1f} FPS")
        
        # Batch processing test
        batch_sizes = [1, 8, 16, 32]
        batch_times = {}
        
        print(f"\n📦 Batch Processing Performance:")
        for batch_size in batch_sizes:
            if batch_size <= len(results['predictions']):
                test_batch = np.random.random((batch_size, *self.cnn_model.model.input_shape[1:]))
                
                # Time batch processing
                start_time = time.time()
                self.cnn_model.model.predict(test_batch, verbose=0)
                end_time = time.time()
                
                batch_time = (end_time - start_time) * 1000
                per_sample_time = batch_time / batch_size
                batch_fps = 1000 / per_sample_time
                
                batch_times[batch_size] = {
                    'total_time': batch_time,
                    'per_sample_time': per_sample_time,
                    'fps': batch_fps
                }
                
                print(f"   Batch size {batch_size}: {per_sample_time:.2f} ms/sample ({batch_fps:.1f} FPS)")
        
        return {
            'model_size': model_size,
            'model_memory_mb': model_memory,
            'single_inference': {
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'fps': fps
            },
            'batch_processing': batch_times
        }
    
    def generate_comprehensive_report(self, save_report=True):
        """Generate comprehensive evaluation report"""
        if not self.evaluation_results:
            print("❌ No evaluation results available")
            return None
        
        print(f"\n📋 Generating Comprehensive Report")
        print("=" * 40)
        
        report = {
            'model_info': {
                'model_path': str(self.model_path),
                'architecture': 'CNN',
                'framework': 'TensorFlow/Keras',
                'classes': self.cnn_model.class_names,
                'parameters': self.cnn_model.model.count_params()
            },
            'evaluation_results': self.evaluation_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if save_report:
            report_file = self.output_dir / 'comprehensive_evaluation_report.json'
            with open(report_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    return obj
                
                json.dump(report, f, indent=2, default=convert_numpy)
            
            print(f"💾 Comprehensive report saved: {report_file}")
        
        return report
    
    def run_full_evaluation(self, test_split='test'):
        """Run complete evaluation suite"""
        print("🫀 RhythmGuard CNN Full Evaluation Suite")
        print("=" * 60)
        
        # Load model
        if not self.load_model():
            return False
        
        # Evaluate test set
        results = self.evaluate_test_set(test_split)
        if results is None:
            return False
        
        # Generate classification report
        self.generate_classification_report(results)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(results, normalize=False)
        self.plot_confusion_matrix(results, normalize=True)
        
        # ROC curves
        self.plot_roc_curves(results)
        
        # Clinical analysis
        self.clinical_performance_analysis(results)
        
        # Performance benchmark
        self.performance_benchmark(results)
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        print(f"\n🎉 Full evaluation completed!")
        print(f"📁 Results saved in: {self.output_dir}")
        
        return True


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive CNN Model Evaluation')
    parser.add_argument('model_path', type=str, help='Path to trained CNN model')
    parser.add_argument('--data_path', type=str, default='.', 
                       help='Path to dataset directory')
    parser.add_argument('--test_split', type=str, default='test',
                       help='Test dataset split name')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = CNNModelEvaluator(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    success = evaluator.run_full_evaluation(args.test_split)
    
    if success:
        print(f"\n✅ Evaluation completed successfully!")
    else:
        print(f"\n❌ Evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    main()