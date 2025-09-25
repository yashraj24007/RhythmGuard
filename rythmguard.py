#!/usr/bin/env python3
"""
🫀 RhythmGuard - AI-Powered ECG Monitoring System
================================================
Main entry point for the RhythmGuard ECG arrhythmia classification system.

RhythmGuard is an AI-powered ECG monitoring system designed to analyze heart 
signals in real time. It processes raw electrocardiogram (ECG) data, detects 
irregularities, and classifies them into 5–6 common arrhythmia types such as:

- Normal (N): Normal sinus rhythm
- Atrial Fibrillation/Supraventricular (S): Atrial premature beats
- Premature Ventricular Contractions (V): PVC
- Fusion beats (F): Fusion of ventricular + normal beat
- Unknown/Paced (Q): Paced beats, unclassifiable beats
- Myocardial Infarction (M): MI markers

In addition to identifying the specific arrhythmia, RhythmGuard also predicts 
the severity level—Mild, Moderate, or Severe—helping healthcare providers 
prioritize urgent cases and enable early intervention.
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import system components with new structure
from src.training.train_cnn import train_cnn_model
from src.evaluation.test_cnn_model import test_cnn_model
from src.evaluation.cnn_full_evaluation import CNNModelEvaluator
from src.preprocessing.cnn_ecg_preprocessor import CNNECGPreprocessor
from src.models.severity_predictor import SeverityPredictor

# For demo functionality
try:
    import sys
    sys.path.append('scripts')
    from cnn_quick_demo import RhythmGuardCNNDemo
except ImportError:
    RhythmGuardCNNDemo = None

def check_dataset(data_path="."):
    """Check if dataset is properly structured"""
    data_path = Path(data_path)
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    
    print("🔍 Checking dataset structure...")
    
    if not train_dir.exists():
        print(f"❌ Training directory not found: {train_dir}")
        return False
    
    if not test_dir.exists():
        print(f"⚠️ Test directory not found: {test_dir}")
        print("💡 Test directory is optional but recommended")
    
    # Check for ECG classes
    expected_classes = ['N', 'S', 'V', 'F', 'Q', 'M']
    found_classes = []
    
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir() and class_dir.name in expected_classes:
            image_count = len(list(class_dir.glob('*.png')))
            found_classes.append(class_dir.name)
            print(f"✅ Found class {class_dir.name}: {image_count} images")
    
    if not found_classes:
        print("❌ No valid ECG class directories found")
        print(f"💡 Expected classes: {expected_classes}")
        return False
    
    print(f"✅ Dataset check complete. Found {len(found_classes)} classes")
    return True

def train_model(args):
    """Train CNN model"""
    print("🚀 Training RhythmGuard CNN Model")
    print("=" * 50)
    
    if not check_dataset(args.data_path):
        return False
    
    # Train model
    cnn, history, summary = train_cnn_model(
        data_path=args.data_path,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        augment_data=not args.no_augmentation
    )
    
    if cnn is not None:
        print("🎉 Training completed successfully!")
        return True
    else:
        print("❌ Training failed!")
        return False

def test_model(args):
    """Test trained model"""
    print("🧪 Testing RhythmGuard CNN Model")
    print("=" * 50)
    
    # Auto-detect latest model if not specified
    model_path = args.model_path
    if not model_path:
        latest_model_file = "models/latest_model.txt"
        if os.path.exists(latest_model_file):
            with open(latest_model_file, 'r') as f:
                model_name = f.read().strip()
            model_path = f"models/{model_name}.keras"
            print(f"🔍 Using latest trained model: {model_path}")
        else:
            print("❌ No trained model found. Please train a model first:")
            print("   python quick_train.py")
            return False
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    success = test_cnn_model(
        model_path=model_path,
        data_path=args.data_path,
        test_split=args.test_split,
        max_samples_per_class=args.max_samples,
        save_results=not args.no_save
    )
    
    return success

def evaluate_model(args):
    """Comprehensive model evaluation"""
    print("📊 Comprehensive Model Evaluation")
    print("=" * 50)
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return False
    
    evaluator = CNNModelEvaluator(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    success = evaluator.run_full_evaluation(args.test_split)
    return success

def demo_model(args):
    """Run model demonstration"""
    print("🎭 RhythmGuard Model Demo")
    print("=" * 50)
    
    demo = RhythmGuardCNNDemo(
        data_path=args.data_path,
        model_path=args.model_path
    )
    
    if args.demo_mode == 'full':
        demo.run_full_demo()
    elif args.demo_mode == 'single':
        if demo.load_model():
            demo.demo_single_prediction()
    elif args.demo_mode == 'batch':
        if demo.load_model():
            demo.demo_batch_prediction()
    elif args.demo_mode == 'augment':
        demo.demo_data_augmentation()
    elif args.demo_mode == 'dataset':
        demo.demo_dataset_analysis()
    
    return True

def analyze_dataset(args):
    """Analyze dataset statistics"""
    print("📊 Dataset Analysis")
    print("=" * 50)
    
    preprocessor = CNNECGPreprocessor(args.data_path)
    
    # Analyze training data
    if (Path(args.data_path) / 'train').exists():
        print("🔍 Analyzing training dataset...")
        train_stats = preprocessor.analyze_dataset('train')
        
        # Visualize samples
        preprocessor.visualize_samples(
            'train', 
            samples_per_class=3,
            save_path='dataset_train_samples.png'
        )
    
    # Analyze test data
    if (Path(args.data_path) / 'test').exists():
        print("\n🔍 Analyzing test dataset...")
        test_stats = preprocessor.analyze_dataset('test')
        
        # Visualize samples
        preprocessor.visualize_samples(
            'test', 
            samples_per_class=3,
            save_path='dataset_test_samples.png'
        )
    
    return True

def predict_single(args):
    """Predict single ECG image"""
    print("🔍 Single ECG Prediction")
    print("=" * 50)
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return False
    
    if not os.path.exists(args.image_path):
        print(f"❌ Image file not found: {args.image_path}")
        return False
    
    # Load model and make prediction
    from src.models.cnn_model import RhythmGuardCNN
    
    cnn_model = RhythmGuardCNN()
    cnn_model.load_model(args.model_path)
    
    # Get prediction
    result = cnn_model.predict_single_image(args.image_path)
    
    # Get severity assessment
    severity_predictor = SeverityPredictor()
    severity_result = severity_predictor.predict_severity_rule_based(
        result['predicted_class']
    )
    
    # Display results
    print(f"📷 Image: {Path(args.image_path).name}")
    print(f"🎯 Predicted Class: {result['predicted_class']}")
    print(f"📊 Confidence: {result['confidence']:.1%}")
    print(f"🏥 Severity: {severity_result['severity']}")
    print(f"📝 Description: {severity_result.get('description', '')}")
    
    print(f"\n📊 All Class Probabilities:")
    for class_name, prob in result['all_probabilities'].items():
        print(f"   {class_name}: {prob:.1%}")
    
    return True

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='RhythmGuard - AI-Powered ECG Monitoring System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rythmguard.py train --epochs 50 --architecture custom
  python rythmguard.py test model.h5 --test_split test
  python rythmguard.py evaluate model.h5 --output_dir results
  python rythmguard.py demo --model_path model.h5 --demo_mode single
  python rythmguard.py predict model.h5 image.png
  python rythmguard.py analyze --data_path ./dataset
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train CNN model')
    train_parser.add_argument('--data_path', type=str, default='.', 
                             help='Path to dataset directory')
    train_parser.add_argument('--architecture', type=str, default='custom',
                             choices=['custom', 'vgg16', 'resnet50'],
                             help='Model architecture')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=32,
                             help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, default=0.001,
                             help='Learning rate')
    train_parser.add_argument('--validation_split', type=float, default=0.2,
                             help='Validation split fraction')
    train_parser.add_argument('--no_augmentation', action='store_true',
                             help='Disable data augmentation')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test trained model')
    test_parser.add_argument('model_path', type=str, help='Path to trained model')
    test_parser.add_argument('--data_path', type=str, default='.',
                            help='Path to dataset directory')
    test_parser.add_argument('--test_split', type=str, default='test',
                            help='Test dataset split name')
    test_parser.add_argument('--max_samples', type=int, default=None,
                            help='Maximum samples per class for testing')
    test_parser.add_argument('--no_save', action='store_true',
                            help='Don\'t save test results')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Comprehensive evaluation')
    eval_parser.add_argument('model_path', type=str, help='Path to trained model')
    eval_parser.add_argument('--data_path', type=str, default='.',
                            help='Path to dataset directory')
    eval_parser.add_argument('--test_split', type=str, default='test',
                            help='Test dataset split name')
    eval_parser.add_argument('--output_dir', type=str, default='evaluation_results',
                            help='Output directory for results')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run model demonstration')
    demo_parser.add_argument('--model_path', type=str, default=None,
                            help='Path to trained model')
    demo_parser.add_argument('--data_path', type=str, default='.',
                            help='Path to dataset directory')
    demo_parser.add_argument('--demo_mode', type=str, default='full',
                            choices=['full', 'single', 'batch', 'augment', 'dataset'],
                            help='Demo mode')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict single ECG image')
    predict_parser.add_argument('model_path', type=str, help='Path to trained model')
    predict_parser.add_argument('image_path', type=str, help='Path to ECG image')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dataset')
    analyze_parser.add_argument('--data_path', type=str, default='.',
                               help='Path to dataset directory')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Set TensorFlow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Execute command
    success = False
    
    if args.command == 'train':
        success = train_model(args)
    elif args.command == 'test':
        success = test_model(args)
    elif args.command == 'evaluate':
        success = evaluate_model(args)
    elif args.command == 'demo':
        success = demo_model(args)
    elif args.command == 'predict':
        success = predict_single(args)
    elif args.command == 'analyze':
        success = analyze_dataset(args)
    
    if success:
        print("\n✅ Command completed successfully!")
    else:
        print("\n❌ Command failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()