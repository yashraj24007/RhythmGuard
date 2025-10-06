#!/usr/bin/env python3
"""
Limited Data Training Script for RhythmGuard
Train with exactly 10 samples from each ECG type for quick testing.
"""

import os
import sys
import json
import shutil
import random
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.train_cnn import train_cnn_model
from src.models.cnn_model import RhythmGuardCNN

def create_limited_dataset(source_dir="data", target_dir="data_limited", samples_per_class=10):
    """
    Create a limited dataset with specified number of samples per class.
    """
    print(f"🔧 Creating limited dataset with {samples_per_class} samples per class...")
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Remove existing limited dataset
    if target_path.exists():
        shutil.rmtree(target_path)
    
    # Create target directories
    for split in ['train', 'test']:
        for class_name in ['F', 'M', 'N', 'Q', 'S', 'V']:
            os.makedirs(target_path / split / class_name, exist_ok=True)
    
    # ECG class descriptions
    class_info = {
        'N': 'Normal sinus rhythm',
        'S': 'Supraventricular arrhythmias', 
        'V': 'Ventricular arrhythmias',
        'F': 'Fusion beats',
        'Q': 'Unknown/Paced beats',
        'M': 'Myocardial infarction'
    }
    
    # Copy limited samples for each class and split
    for split in ['train', 'test']:
        print(f"\n📂 Processing {split} split:")
        split_samples = samples_per_class if split == 'train' else max(1, samples_per_class // 2)
        
        for class_name in ['F', 'M', 'N', 'Q', 'S', 'V']:
            source_class_dir = source_path / split / class_name
            target_class_dir = target_path / split / class_name
            
            if source_class_dir.exists():
                # Get all image files
                all_files = list(source_class_dir.glob('*.png'))
                
                if len(all_files) >= split_samples:
                    # Randomly select samples
                    selected_files = random.sample(all_files, split_samples)
                    
                    # Copy selected files
                    for file_path in selected_files:
                        shutil.copy2(file_path, target_class_dir / file_path.name)
                    
                    print(f"  ✅ {class_name} ({class_info[class_name]}): {split_samples} samples")
                else:
                    print(f"  ⚠️  {class_name}: Only {len(all_files)} samples available (needed {split_samples})")
                    # Copy all available files
                    for file_path in all_files:
                        shutil.copy2(file_path, target_class_dir / file_path.name)
            else:
                print(f"  ❌ {class_name}: Directory not found")
    
    return str(target_path)

def train_with_limited_data():
    """
    Train the model with limited data for quick testing.
    """
    print("🚀 RhythmGuard Limited Data Training")
    print("=" * 60)
    
    # Create limited dataset
    limited_data_path = create_limited_dataset(
        source_dir="data", 
        target_dir="data_limited", 
        samples_per_class=10
    )
    
    # Training configuration optimized for small dataset
    config = {
        'data_path': limited_data_path,
        'epochs': 20,  # Fewer epochs for small dataset
        'batch_size': 8,   # Smaller batch size
        'architecture': 'custom',
        'learning_rate': 0.001,
        'validation_split': 0.3,  # Higher validation split for small data
        'augment_data': True  # Important for small dataset
    }
    
    print(f"\n📊 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Train the model
    try:
        model, history, report = train_cnn_model(
            data_path=config['data_path'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            architecture=config['architecture'],
            learning_rate=config['learning_rate'],
            validation_split=config['validation_split'],
            augment_data=config['augment_data']
        )
        
        # Create timestamp for unique model naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"rhythmguard_cnn_limited_{config['architecture']}_{timestamp}"
        
        # Save to multiple locations for redundancy
        model_paths = [
            f"models/{model_name}.keras",
            f"rythmguard_output/models/{model_name}.keras"
        ]
        
        # Ensure directories exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("rythmguard_output/models", exist_ok=True)
        
        # Save model properly using the CNN class method
        for model_path in model_paths:
            # Get the underlying Keras model and save it
            keras_model = model.model  # Get the actual Keras model from RhythmGuardCNN
            keras_model.save(model_path)
            print(f"✅ Model saved to: {model_path}")
        
        # Save training configuration and results
        results = {
            'model_name': model_name,
            'timestamp': timestamp,
            'config': config,
            'training_type': 'limited_data',
            'samples_per_class': 10,
            'final_accuracy': float(max(history.history.get('val_accuracy', [0]))),
            'final_loss': float(min(history.history.get('val_loss', [1]))),
            'training_completed': True
        }
        
        # Save results metadata
        with open(f"models/{model_name}_info.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        with open(f"rythmguard_output/models/{model_name}_info.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n🎉 Limited Data Training completed successfully!")
        print(f"📁 Model stored as: {model_name}")
        print(f"🎯 Final Validation Accuracy: {results['final_accuracy']:.4f}")
        print(f"📉 Final Validation Loss: {results['final_loss']:.4f}")
        
        # Create a 'latest_model.txt' file pointing to the most recent model
        with open("models/latest_model.txt", 'w') as f:
            f.write(model_name)
        with open("rythmguard_output/models/latest_model.txt", 'w') as f:
            f.write(model_name)
        
        print(f"\n💡 Dataset used: 10 samples per ECG type")
        print("🧪 Now you can test with:")
        print(f"   py rythmguard.py test")
        print("   py rythmguard.py predict --image data/test/N/N123.png")
        
        # Clean up limited dataset (optional)
        print(f"\n🧹 Cleaning up temporary dataset: {limited_data_path}")
        try:
            shutil.rmtree(limited_data_path)
            print("✅ Temporary dataset removed")
        except:
            print("⚠️  Could not remove temporary dataset")
        
        return model_name, results
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("Please check your dataset structure and requirements.")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    train_with_limited_data()