#!/usr/bin/env python3
"""
🫀 RhythmGuard Setup Verification
================================
Verify that all dependencies are installed and the system is ready to use.
"""

import sys
import importlib
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("\n📦 Checking dependencies...")
    
    required_packages = {
        'tensorflow': 'TensorFlow',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'albumentations': 'Albumentations',
        'joblib': 'Joblib'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - Missing")
            missing_packages.append(name)
    
    return len(missing_packages) == 0

def check_file_structure():
    """Check file structure"""
    print("\n📁 Checking file structure...")
    
    required_files = [
        'rythmguard.py',
        'cnn_model.py',
        'cnn_ecg_preprocessor.py',
        'train_cnn.py',
        'test_cnn_model.py',
        'cnn_full_evaluation.py',
        'cnn_ecg_augmentor.py',
        'cnn_quick_demo.py',
        'severity_predictor.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - OK")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_dataset_structure():
    """Check dataset structure"""
    print("\n📊 Checking dataset structure...")
    
    train_dir = Path("train")
    test_dir = Path("test")
    
    if not train_dir.exists():
        print("❌ Training directory 'train/' not found")
        print("💡 Create train/ directory with ECG class subdirectories")
        return False
    
    print("✅ Training directory found")
    
    # Check for ECG classes
    expected_classes = ['N', 'S', 'V', 'F', 'Q', 'M']
    found_classes = []
    
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir() and class_dir.name in expected_classes:
            image_count = len(list(class_dir.glob('*.png')))
            found_classes.append(class_dir.name)
            print(f"✅ Class {class_dir.name}: {image_count} images")
    
    if not found_classes:
        print("⚠️ No ECG class directories found in train/")
        print(f"💡 Expected classes: {expected_classes}")
        return False
    
    if test_dir.exists():
        print("✅ Test directory found")
    else:
        print("⚠️ Test directory 'test/' not found (optional)")
    
    return True

def check_gpu_support():
    """Check GPU support"""
    print("\n🔥 Checking GPU support...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU support available: {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print("⚠️ No GPU detected - will use CPU")
            print("💡 Install CUDA for GPU acceleration")
        return True
    except ImportError:
        print("❌ TensorFlow not available")
        return False

def run_quick_test():
    """Run a quick system test"""
    print("\n🧪 Running quick system test...")
    
    try:
        # Test imports
        sys.path.append('.')
        from cnn_model import RhythmGuardCNN
        from cnn_ecg_preprocessor import CNNECGPreprocessor
        from severity_predictor import SeverityPredictor
        
        print("✅ All modules import successfully")
        
        # Test basic functionality
        preprocessor = CNNECGPreprocessor(".")
        severity_predictor = SeverityPredictor()
        
        # Test severity prediction
        test_result = severity_predictor.predict_severity_rule_based('N')
        print(f"✅ Severity prediction test: {test_result['severity']}")
        
        print("✅ Quick system test passed")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    """Main setup verification function"""
    print("🫀 RhythmGuard Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("File Structure", check_file_structure),
        ("Dataset Structure", check_dataset_structure),
        ("GPU Support", check_gpu_support),
        ("System Test", run_quick_test)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed_checks += 1
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
    
    print(f"\n📋 Setup Verification Summary")
    print("=" * 30)
    print(f"Passed: {passed_checks}/{total_checks} checks")
    
    if passed_checks == total_checks:
        print("🎉 All checks passed! RhythmGuard is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Prepare your ECG dataset in train/ and test/ directories")
        print("   2. Run: python rythmguard.py analyze")
        print("   3. Train model: python rythmguard.py train")
        print("   4. Test model: python rythmguard.py test model.h5")
    else:
        print("⚠️ Some checks failed. Please resolve the issues above.")
        print("\n🔧 Common solutions:")
        print("   - Install missing packages: pip install -r requirements.txt")
        print("   - Organize dataset into proper directory structure")
        print("   - Ensure Python 3.8+ is installed")
    
    return passed_checks == total_checks

if __name__ == "__main__":
    main()