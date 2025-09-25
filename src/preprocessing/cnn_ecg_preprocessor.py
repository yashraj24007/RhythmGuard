"""
🫀 RhythmGuard CNN ECG Preprocessor
==================================
Enhanced ECG image preprocessing pipeline optimized for CNN models.
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import albumentations as A
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CNNECGPreprocessor:
    """
    CNN-optimized ECG Image Preprocessing Class for RythmGuard System
    """
    
    def __init__(self, data_path, target_size=(224, 224), channels=3):
        """
        Initialize the CNN ECG Preprocessor
        
        Args:
            data_path (str): Path to the dataset directory
            target_size (tuple): Target size for image resizing (height, width)
            channels (int): Number of color channels (1 for grayscale, 3 for RGB)
        """
        self.data_path = Path(data_path)
        self.target_size = target_size
        self.channels = channels
        self.input_shape = (*target_size, channels)
        
        # ECG Classification mapping
        self.class_mapping = {
            'N': {
                'name': 'Normal', 
                'description': 'Normal beat (sinus rhythm, bundle branch block, etc.)',
                'severity': 'Low',
                'color': '#2E8B57'
            },
            'S': {
                'name': 'Supraventricular', 
                'description': 'Atrial premature beats, supraventricular ectopics',
                'severity': 'Medium',
                'color': '#FF8C00'
            },
            'V': {
                'name': 'Ventricular', 
                'description': 'PVC (Premature Ventricular Contractions)',
                'severity': 'High',
                'color': '#DC143C'
            },
            'F': {
                'name': 'Fusion', 
                'description': 'Fusion of ventricular + normal beat',
                'severity': 'Medium',
                'color': '#9932CC'
            },
            'Q': {
                'name': 'Unknown', 
                'description': 'Paced beats, unclassifiable beats',
                'severity': 'Medium',
                'color': '#708090'
            },
            'M': {
                'name': 'Myocardial Infarction', 
                'description': 'MI - Myocardial Infarction markers',
                'severity': 'Critical',
                'color': '#8B0000'
            }
        }
        
        self.label_encoder = LabelEncoder()
        self.class_names = list(self.class_mapping.keys())
        
        # CNN-specific augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline()
        
    def _create_augmentation_pipeline(self):
        """Create albumentations augmentation pipeline optimized for ECG images"""
        return A.Compose([
            # Geometric transformations (mild for ECG signals)
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=5, 
                p=0.3
            ),
            
            # Brightness and contrast (important for ECG signal clarity)
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.3
            ),
            
            # Noise addition (simulate real-world ECG noise)
            A.GaussNoise(
                var_limit=(10.0, 50.0), 
                p=0.2
            ),
            
            # Elastic transform (very mild for ECG)
            A.ElasticTransform(
                alpha=50, 
                sigma=5, 
                alpha_affine=5, 
                p=0.1
            ),
            
            # Grid distortion (very mild)
            A.GridDistortion(
                num_steps=3, 
                distort_limit=0.1, 
                p=0.1
            ),
            
            # Blur (simulate motion artifacts)
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5)
            ], p=0.1),
            
            # Normalize to [0, 1] range
            A.Normalize(
                mean=[0.485, 0.456, 0.406] if self.channels == 3 else [0.5],
                std=[0.229, 0.224, 0.225] if self.channels == 3 else [0.5],
                max_pixel_value=255.0
            )
        ])
    
    def load_and_preprocess_image(self, image_path, apply_augmentation=False):
        """
        Load and preprocess a single ECG image for CNN input
        
        Args:
            image_path (str): Path to the image file
            apply_augmentation (bool): Whether to apply data augmentation
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        try:
            # Load image
            if self.channels == 1:
                # Grayscale
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    return None
                image = cv2.resize(image, self.target_size)
                image = np.expand_dims(image, axis=-1)
            else:
                # RGB
                image = cv2.imread(str(image_path))
                if image is None:
                    return None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.target_size)
            
            # Apply augmentation if requested
            if apply_augmentation:
                augmented = self.augmentation_pipeline(image=image)
                image = augmented['image']
            else:
                # Basic normalization without augmentation
                image = image.astype(np.float32) / 255.0
                if self.channels == 3:
                    # Apply ImageNet normalization
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image = (image - mean) / std
                else:
                    # Simple normalization for grayscale
                    image = (image - 0.5) / 0.5
            
            return image
            
        except Exception as e:
            print(f"❌ Error processing image {image_path}: {e}")
            return None
    
    def create_cnn_dataset(self, split='train', max_samples_per_class=None, 
                          apply_augmentation=True, return_generators=False):
        """
        Create CNN-ready dataset
        
        Args:
            split (str): Dataset split ('train', 'test', 'val')
            max_samples_per_class (int): Maximum samples per class (None for all)
            apply_augmentation (bool): Whether to apply augmentation
            return_generators (bool): Return TensorFlow data generators
            
        Returns:
            tuple: (X, y, class_names) or data generators
        """
        split_path = self.data_path / split
        
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset split not found: {split_path}")
        
        print(f"📁 Loading {split} dataset from {split_path}")
        
        X, y, image_paths = [], [], []
        class_folders = sorted([f for f in split_path.iterdir() if f.is_dir()])
        
        for class_idx, class_folder in enumerate(class_folders):
            class_name = class_folder.name
            
            if class_name not in self.class_mapping:
                print(f"⚠️ Unknown class: {class_name}, skipping...")
                continue
            
            # Get image files
            image_files = list(class_folder.glob('*.png'))
            
            if max_samples_per_class:
                image_files = image_files[:max_samples_per_class]
            
            print(f"📊 Processing {class_name}: {len(image_files)} images")
            
            for img_path in image_files:
                # Load and preprocess image
                processed_img = self.load_and_preprocess_image(
                    img_path, 
                    apply_augmentation=apply_augmentation and split == 'train'
                )
                
                if processed_img is not None:
                    X.append(processed_img)
                    y.append(class_idx)
                    image_paths.append(str(img_path))
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Get class names in order
        class_names = [folder.name for folder in class_folders 
                      if folder.name in self.class_mapping]
        
        print(f"✅ Dataset created: {X.shape[0]} samples, {len(class_names)} classes")
        print(f"📊 Input shape: {X.shape}")
        print(f"🎯 Classes: {class_names}")
        
        if return_generators:
            return self._create_tf_dataset(X, y, class_names, split)
        
        return X, y, class_names, image_paths
    
    def _create_tf_dataset(self, X, y, class_names, split):
        """Create TensorFlow dataset from arrays"""
        # Convert labels to categorical
        num_classes = len(class_names)
        y_categorical = tf.keras.utils.to_categorical(y, num_classes)
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y_categorical))
        
        if split == 'train':
            dataset = dataset.shuffle(buffer_size=1000)
        
        return dataset
    
    def analyze_dataset(self, split='train'):
        """Analyze dataset statistics"""
        split_path = self.data_path / split
        
        if not split_path.exists():
            print(f"❌ Dataset split not found: {split_path}")
            return None
        
        print(f"📊 Analyzing {split} dataset...")
        
        class_stats = {}
        total_images = 0
        
        class_folders = sorted([f for f in split_path.iterdir() if f.is_dir()])
        
        for class_folder in class_folders:
            class_name = class_folder.name
            
            if class_name not in self.class_mapping:
                continue
            
            # Count images
            image_files = list(class_folder.glob('*.png'))
            count = len(image_files)
            
            # Sample image for size analysis
            if image_files:
                sample_img = cv2.imread(str(image_files[0]))
                original_shape = sample_img.shape if sample_img is not None else (0, 0, 0)
            else:
                original_shape = (0, 0, 0)
            
            class_stats[class_name] = {
                'count': count,
                'percentage': 0,  # Will calculate after total
                'original_shape': original_shape,
                'info': self.class_mapping[class_name]
            }
            
            total_images += count
        
        # Calculate percentages
        for class_name in class_stats:
            class_stats[class_name]['percentage'] = (
                class_stats[class_name]['count'] / total_images * 100
            )
        
        # Print analysis
        print(f"\n📈 Dataset Analysis - {split.upper()} Split")
        print("=" * 60)
        print(f"Total Images: {total_images}")
        print(f"Number of Classes: {len(class_stats)}")
        print(f"Target Size: {self.target_size}")
        print(f"Input Shape: {self.input_shape}")
        
        print(f"\n📊 Class Distribution:")
        for class_name, stats in class_stats.items():
            info = stats['info']
            print(f"  {class_name} ({info['name']}):")
            print(f"    📁 Images: {stats['count']} ({stats['percentage']:.1f}%)")
            print(f"    🏥 Severity: {info['severity']}")
            print(f"    📝 Description: {info['description']}")
            print()
        
        # Check class balance
        counts = [stats['count'] for stats in class_stats.values()]
        min_count, max_count = min(counts), max(counts)
        balance_ratio = min_count / max_count if max_count > 0 else 0
        
        print(f"⚖️ Class Balance Analysis:")
        print(f"  Min samples per class: {min_count}")
        print(f"  Max samples per class: {max_count}")
        print(f"  Balance ratio: {balance_ratio:.3f}")
        
        if balance_ratio < 0.5:
            print(f"  ⚠️ Dataset is imbalanced (ratio < 0.5)")
            print(f"  💡 Consider data augmentation or class weighting")
        else:
            print(f"  ✅ Dataset is reasonably balanced")
        
        return class_stats
    
    def visualize_samples(self, split='train', samples_per_class=3, save_path=None):
        """Visualize sample images from each class"""
        split_path = self.data_path / split
        
        if not split_path.exists():
            print(f"❌ Dataset split not found: {split_path}")
            return
        
        class_folders = sorted([f for f in split_path.iterdir() if f.is_dir() 
                              if f.name in self.class_mapping])
        
        if not class_folders:
            print(f"❌ No valid class folders found")
            return
        
        # Create subplot grid
        rows = len(class_folders)
        cols = samples_per_class
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        for row, class_folder in enumerate(class_folders):
            class_name = class_folder.name
            class_info = self.class_mapping[class_name]
            
            # Get sample images
            image_files = list(class_folder.glob('*.png'))[:samples_per_class]
            
            for col, img_path in enumerate(image_files):
                if col >= cols:
                    break
                
                # Load original image for visualization
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.target_size)
                    
                    axes[row, col].imshow(img)
                    axes[row, col].set_title(f"{class_name} - {class_info['name']}")
                    axes[row, col].axis('off')
                else:
                    axes[row, col].text(0.5, 0.5, 'Image\nNot Found', 
                                       ha='center', va='center')
                    axes[row, col].set_title(f"{class_name} - Error")
                    axes[row, col].axis('off')
            
            # Fill empty columns
            for col in range(len(image_files), cols):
                axes[row, col].axis('off')
        
        plt.suptitle(f'ECG Sample Images - {split.upper()} Dataset', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Sample visualization saved: {save_path}")
        
        plt.show()
    
    def prepare_for_training(self, train_split='train', test_split='test', 
                           validation_split=0.2, max_samples_per_class=None):
        """
        Prepare complete dataset for CNN training
        
        Args:
            train_split (str): Training data folder name
            test_split (str): Test data folder name  
            validation_split (float): Fraction for validation split
            max_samples_per_class (int): Maximum samples per class
            
        Returns:
            tuple: Training and validation data generators
        """
        print(f"🔄 Preparing dataset for CNN training...")
        
        # Load training data
        X_train, y_train, class_names, _ = self.create_cnn_dataset(
            split=train_split,
            max_samples_per_class=max_samples_per_class,
            apply_augmentation=False  # Will handle augmentation in generators
        )
        
        # Load test data if available
        test_data = None
        test_path = self.data_path / test_split
        if test_path.exists():
            X_test, y_test, _, _ = self.create_cnn_dataset(
                split=test_split,
                apply_augmentation=False
            )
            test_data = (X_test, y_test)
        
        # Split training data for validation
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, 
                test_size=validation_split, 
                stratify=y_train,
                random_state=42
            )
            val_data = (X_val, y_val)
        else:
            val_data = None
        
        print(f"✅ Dataset prepared:")
        print(f"   📁 Training samples: {X_train.shape[0]}")
        if val_data:
            print(f"   📁 Validation samples: {X_val.shape[0]}")
        if test_data:
            print(f"   📁 Test samples: {X_test.shape[0]}")
        print(f"   🎯 Classes: {len(class_names)}")
        print(f"   📊 Input shape: {X_train.shape[1:]}")
        
        return {
            'train_data': (X_train, y_train),
            'val_data': val_data,
            'test_data': test_data,
            'class_names': class_names,
            'input_shape': X_train.shape[1:]
        }
    
    def get_class_weights(self, y):
        """Calculate class weights for imbalanced datasets"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y
        )
        
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"📊 Calculated class weights:")
        for class_idx, weight in class_weight_dict.items():
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"Class_{class_idx}"
            print(f"   {class_name}: {weight:.3f}")
        
        return class_weight_dict