#!/usr/bin/env python3
"""
Model Status Checker for RhythmGuard
Check if models are trained and stored.
"""

import os
import json
import glob
from datetime import datetime

def check_model_status():
    """Check the status of trained models"""
    print("🔍 RhythmGuard Model Status Check")
    print("=" * 50)
    
    # Check for model files
    model_locations = [
        "models/",
        "rythmguard_output/models/"
    ]
    
    trained_models = []
    
    for location in model_locations:
        if os.path.exists(location):
            # Check for .keras files
            keras_files = glob.glob(os.path.join(location, "*.keras"))
            h5_files = glob.glob(os.path.join(location, "*.h5"))
            
            all_models = keras_files + h5_files
            
            if all_models:
                print(f"📁 Found models in {location}:")
                for model_file in all_models:
                    model_name = os.path.basename(model_file)
                    size_mb = os.path.getsize(model_file) / (1024 * 1024)
                    
                    # Check for corresponding info file
                    info_file = model_file.replace('.keras', '_info.json').replace('.h5', '_info.json')
                    
                    print(f"  ✅ {model_name} ({size_mb:.1f} MB)")
                    
                    if os.path.exists(info_file):
                        try:
                            with open(info_file, 'r') as f:
                                info = json.load(f)
                            print(f"     📊 Accuracy: {info.get('final_accuracy', 'N/A')}")
                            print(f"     📅 Trained: {info.get('timestamp', 'N/A')}")
                        except:
                            print("     ⚠️  No training info available")
                    
                    trained_models.append(model_file)
                print()
            else:
                print(f"📁 {location}: No models found")
        else:
            print(f"📁 {location}: Directory doesn't exist")
    
    # Check for latest model pointer
    latest_files = [
        "models/latest_model.txt",
        "rythmguard_output/models/latest_model.txt"
    ]
    
    for latest_file in latest_files:
        if os.path.exists(latest_file):
            with open(latest_file, 'r') as f:
                latest_model = f.read().strip()
            print(f"🎯 Latest model pointer ({latest_file}): {latest_model}")
    
    print("\n" + "=" * 50)
    
    if trained_models:
        print(f"✅ Status: {len(trained_models)} trained model(s) found")
        print("\n💡 Ready to test with:")
        print("   python rythmguard.py test")
        print("   python rythmguard.py predict --image path/to/ecg.png")
        return True
    else:
        print("❌ Status: No trained models found")
        print("\n💡 To train your model:")
        print("   python quick_train.py")
        print("\n   or use the full training:")
        print("   python rythmguard.py train --epochs 30")
        return False

if __name__ == "__main__":
    check_model_status()