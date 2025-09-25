# 🫀 RhythmGuard Project Structure

## 📁 Directory Organization

```
RhythmGuard/
├── 📄 rythmguard.py           # Main CLI entry point
├── 📄 requirements.txt        # Python dependencies
├── 📄 README.md              # Project documentation
├── 📄 LICENSE                # MIT License
├── 📄 .gitignore            # Git ignore patterns
│
├── 📁 src/                   # Source code modules
│   ├── 📄 __init__.py
│   │
│   ├── 📁 models/           # CNN models and predictors
│   │   ├── 📄 __init__.py
│   │   ├── 📄 cnn_model.py          # CNN architectures (Custom, VGG16, ResNet50)
│   │   └── 📄 severity_predictor.py  # Severity level prediction
│   │
│   ├── 📁 preprocessing/    # Data preprocessing and augmentation
│   │   ├── 📄 __init__.py
│   │   ├── 📄 cnn_ecg_preprocessor.py  # ECG image preprocessing
│   │   └── 📄 cnn_ecg_augmentor.py     # Data augmentation pipeline
│   │
│   ├── 📁 training/         # Model training modules
│   │   ├── 📄 __init__.py
│   │   └── 📄 train_cnn.py          # CNN training pipeline
│   │
│   ├── 📁 evaluation/       # Model testing and evaluation
│   │   ├── 📄 __init__.py
│   │   ├── 📄 test_cnn_model.py     # Model testing framework
│   │   └── 📄 cnn_full_evaluation.py # Comprehensive evaluation suite
│   │
│   └── 📁 utils/           # Utility functions and tools
│       ├── 📄 __init__.py
│       ├── 📄 setup_check.py        # Environment setup verification
│       └── 📄 check_models.py       # Model status checker
│
├── 📁 scripts/             # Standalone scripts and demos
│   ├── 📄 quick_train.py            # Quick model training script
│   └── 📄 cnn_quick_demo.py         # Interactive demos
│
├── 📁 configs/             # Configuration files
│   └── 📄 config.ini               # System configuration
│
├── 📁 data/               # Dataset directory
│   ├── 📁 train/                   # Training ECG images
│   │   ├── 📁 F/  # Fusion beats
│   │   ├── 📁 M/  # Myocardial Infarction
│   │   ├── 📁 N/  # Normal sinus rhythm
│   │   ├── 📁 Q/  # Unknown/Paced beats
│   │   ├── 📁 S/  # Supraventricular
│   │   └── 📁 V/  # Ventricular
│   │
│   └── 📁 test/                    # Testing ECG images
│       ├── 📁 F/  # Fusion beats
│       ├── 📁 M/  # Myocardial Infarction
│       ├── 📁 N/  # Normal sinus rhythm
│       ├── 📁 Q/  # Unknown/Paced beats
│       ├── 📁 S/  # Supraventricular
│       └── 📁 V/  # Ventricular
│
├── 📁 models/             # Trained model storage
│   └── 📄 latest_model.txt         # Pointer to latest trained model
│
├── 📁 rythmguard_output/  # Output directory
│   ├── 📁 models/                  # Backup model storage
│   ├── 📁 reports/                 # Evaluation reports
│   └── 📁 visualizations/          # Training plots and analysis
│
└── 📁 docs/               # Documentation
    └── 📄 project_structure.md     # This file
```

## 🚀 Usage Patterns

### Quick Start
```bash
# Train model once
python scripts/quick_train.py

# Test repeatedly
python rythmguard.py test

# Check status
python src/utils/check_models.py
```

### Development Workflow
```bash
# Full training with custom settings
python rythmguard.py train --epochs 50 --architecture custom

# Comprehensive evaluation
python rythmguard.py evaluate --model models/latest.keras

# Interactive demo
python rythmguard.py demo --mode single
```

## 📊 Module Dependencies

```
rythmguard.py (main)
    ├── src.training.train_cnn
    ├── src.evaluation.test_cnn_model
    ├── src.evaluation.cnn_full_evaluation
    └── src.models.severity_predictor

src.training.train_cnn
    ├── src.models.cnn_model
    └── src.models.severity_predictor

src.evaluation.test_cnn_model
    ├── src.models.cnn_model
    ├── src.preprocessing.cnn_ecg_preprocessor
    └── src.models.severity_predictor

scripts.quick_train
    ├── src.training.train_cnn
    └── src.models.cnn_model
```

## 🔧 Key Features

- **Modular Design**: Clean separation of concerns
- **Easy Imports**: Proper package structure with __init__.py files
- **Scalable**: Easy to add new models, preprocessing steps, or evaluation metrics
- **Professional**: Industry-standard Python package layout
- **Version Control**: Organized structure for collaborative development