#!/usr/bin/env python3
"""
Quick Model Training Script for RhythmGuard
This script trains and saves the model for permanent storage.
"""

import os
import sys
import json
from datetime import datetime
from train_cnn import train_cnn_model
from cnn_model import RhythmGuardCNN

def quick_train_and_save():
    """
    Train the model once and save it for future testing.
    """
    print("🚀 Starting RhythmGuard CNN Training...")
    print("=" * 60)
    
    # Training configuration
    config = {
        'data_path': '.',  # Current directory contains train/ and test/ folders
        'epochs': 30,  # Reduced for quicker training
        'batch_size': 32,
        'architecture': 'custom',
        'learning_rate': 0.001,
        'validation_split': 0.2,
        'augment_data': True
    }
    
    print(f"📊 Training Configuration:")
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
        model_name = f"rhythmguard_cnn_{config['architecture']}_{timestamp}"
        
        # Save to multiple locations for redundancy
        model_paths = [
            f"models/{model_name}.keras",
            f"rythmguard_output/models/{model_name}.keras"
        ]
        
        # Save model in both locations
        for model_path in model_paths:
            model.save(model_path)
            print(f"✅ Model saved to: {model_path}")
        
        # Save training configuration and results
        results = {
            'model_name': model_name,
            'timestamp': timestamp,
            'config': config,
            'final_accuracy': float(max(history.history.get('val_accuracy', [0]))),
            'final_loss': float(min(history.history.get('val_loss', [1]))),
            'training_completed': True
        }
        
        # Save results metadata
        with open(f"models/{model_name}_info.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        with open(f"rythmguard_output/models/{model_name}_info.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Model stored as: {model_name}")
        print(f"🎯 Final Validation Accuracy: {results['final_accuracy']:.4f}")
        print(f"📉 Final Validation Loss: {results['final_loss']:.4f}")
        
        # Create a 'latest_model.txt' file pointing to the most recent model
        with open("models/latest_model.txt", 'w') as f:
            f.write(model_name)
        with open("rythmguard_output/models/latest_model.txt", 'w') as f:
            f.write(model_name)
        
        print("\n💡 Now you can test with:")
        print(f"   python rythmguard.py test --model models/{model_name}.keras")
        print("   or")
        print("   python rythmguard.py predict --image path/to/ecg.png")
        
        return model_name, results
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("Please check your dataset structure and requirements.")
        return None, None

if __name__ == "__main__":
    quick_train_and_save()