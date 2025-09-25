#!/usr/bin/env python3
"""
🫀 RhythmGuard CNN Training Script
=================================
Train a CNN model for ECG rhythm classification using TensorFlow/Keras.
"""

import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.cnn_model import RhythmGuardCNN
from src.models.severity_predictor import SeverityPredictor

def train_cnn_model(data_path=".", architecture="custom", epochs=50, batch_size=32, 
                   learning_rate=0.001, validation_split=0.2, augment_data=True):
    """
    Train CNN model for ECG rhythm classification
    
    Args:
        data_path (str): Path to dataset
        architecture (str): Model architecture ('custom', 'vgg16', 'resnet50')
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        validation_split (float): Fraction for validation split
        augment_data (bool): Whether to apply data augmentation
    """
    
    print("🫀 RhythmGuard CNN Training")
    print("=" * 60)
    
    # Check GPU availability
    print(f"🔍 Checking GPU availability...")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU available: {len(gpus)} GPU(s) detected")
        for gpu in gpus:
            print(f"   - {gpu}")
        # Enable memory growth to avoid taking all GPU memory
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
    else:
        print("⚠️ No GPU detected, using CPU")
    
    # Set up paths
    data_path = Path(data_path)
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    
    print(f"📁 Dataset path: {data_path}")
    print(f"📁 Training data: {train_dir}")
    print(f"📁 Test data: {test_dir}")
    
    # Count classes and samples
    class_folders = [f for f in train_dir.iterdir() if f.is_dir()]
    num_classes = len(class_folders)
    
    print(f"🎯 Found {num_classes} classes: {[f.name for f in class_folders]}")
    
    # Count total samples
    total_train_samples = sum(len(list(class_folder.glob('*.png'))) 
                             for class_folder in class_folders)
    print(f"📊 Total training samples: {total_train_samples}")
    
    # Initialize CNN model
    print(f"\n🏗️ Building {architecture} CNN model...")
    cnn = RhythmGuardCNN(
        input_shape=(224, 224, 3),
        num_classes=num_classes,
        model_name=f"rythmguard_cnn_{architecture}"
    )
    
    # Build model
    model = cnn.build_model(architecture=architecture)
    
    # Print model summary
    print(f"\n📋 Model Architecture Summary:")
    model.summary()
    
    # Compile model
    cnn.compile_model(learning_rate=learning_rate, optimizer='adam')
    
    # Create data generators
    print(f"\n📊 Creating data generators...")
    train_gen, val_gen, test_gen = cnn.create_data_generators(
        train_dir=str(train_dir),
        test_dir=str(test_dir) if test_dir.exists() else None,
        validation_split=validation_split,
        batch_size=batch_size,
        augment_data=augment_data
    )
    
    # Calculate steps per epoch
    steps_per_epoch = train_gen.samples // batch_size
    validation_steps = val_gen.samples // batch_size
    
    print(f"🔄 Training configuration:")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Steps per epoch: {steps_per_epoch}")
    print(f"   - Validation steps: {validation_steps}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Data augmentation: {augment_data}")
    
    # Train model
    print(f"\n🚀 Starting training...")
    start_time = time.time()
    
    try:
        history = cnn.train_model(
            train_generator=train_gen,
            validation_generator=val_gen,
            epochs=epochs,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        
        # Plot training history
        print(f"\n📈 Plotting training history...")
        cnn.plot_training_history()
        
        # Evaluate on test set if available
        if test_gen is not None:
            print(f"\n🧪 Evaluating model on test set...")
            test_accuracy, classification_report, confusion_matrix = cnn.evaluate_model(test_gen)
            
            print(f"🎯 Final Test Accuracy: {test_accuracy:.4f}")
        
        # Save model
        print(f"\n💾 Saving trained model...")
        model_filename = f"rythmguard_cnn_{architecture}.h5"
        cnn.save_model(model_filename)
        
        # Save training summary
        training_summary = {
            'architecture': architecture,
            'epochs_trained': len(history.history['loss']),
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'training_time_seconds': training_time,
            'total_parameters': model.count_params(),
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'validation_split': validation_split,
            'data_augmentation': augment_data,
            'class_names': cnn.class_names
        }
        
        if test_gen is not None:
            training_summary['test_accuracy'] = test_accuracy
        
        import joblib
        joblib.dump(training_summary, f"rythmguard_cnn_{architecture}_summary.joblib")
        
        print(f"\n🎉 Training Summary:")
        print(f"   ✅ Architecture: {architecture}")
        print(f"   ✅ Epochs trained: {len(history.history['loss'])}")
        print(f"   ✅ Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"   ✅ Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        if test_gen is not None:
            print(f"   ✅ Test accuracy: {test_accuracy:.4f}")
        print(f"   ✅ Training time: {training_time:.2f} seconds")
        print(f"   ✅ Total parameters: {model.count_params():,}")
        print(f"   ✅ Model saved: {model_filename}")
        
        return cnn, history, training_summary
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        return None, None, None
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise e

def main():
    """Main training function with command line arguments"""
    parser = argparse.ArgumentParser(description='Train RhythmGuard CNN Model')
    parser.add_argument('--data_path', type=str, default='.', 
                       help='Path to dataset directory')
    parser.add_argument('--architecture', type=str, default='custom',
                       choices=['custom', 'vgg16', 'resnet50'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Fraction of training data for validation')
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Disable data augmentation')
    
    args = parser.parse_args()
    
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
        print(f"\n🎉 CNN training completed successfully!")
        print(f"✅ Model ready for deployment")
    else:
        print(f"\n❌ CNN training failed")

if __name__ == "__main__":
    # Set TensorFlow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Run training
    main()