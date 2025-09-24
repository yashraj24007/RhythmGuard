#!/usr/bin/env python3
"""
🫀 RhythmGuard CNN Data Augmentation
===================================
Advanced data augmentation techniques for ECG image datasets using CNN-optimized approaches.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from pathlib import Path
import random
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

class ECGAugmentor:
    """
    CNN-optimized ECG Image Augmentation Class
    """
    
    def __init__(self, target_size=(224, 224), preserve_signal_integrity=True):
        """
        Initialize ECG Augmentor
        
        Args:
            target_size (tuple): Target image size (height, width)
            preserve_signal_integrity (bool): Whether to preserve ECG signal characteristics
        """
        self.target_size = target_size
        self.preserve_signal_integrity = preserve_signal_integrity
        
        # Create different augmentation pipelines
        self.mild_pipeline = self._create_mild_pipeline()
        self.moderate_pipeline = self._create_moderate_pipeline()
        self.aggressive_pipeline = self._create_aggressive_pipeline()
        self.clinical_pipeline = self._create_clinical_pipeline()
        
    def _create_mild_pipeline(self):
        """Create mild augmentation pipeline for preserving signal quality"""
        return A.Compose([
            # Very minimal geometric changes
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=2,
                p=0.3
            ),
            
            # Slight brightness/contrast adjustments
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            
            # Minimal noise
            A.GaussNoise(
                var_limit=(5.0, 15.0),
                p=0.2
            ),
            
            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            )
        ])
    
    def _create_moderate_pipeline(self):
        """Create moderate augmentation pipeline"""
        return A.Compose([
            # Geometric transformations
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=5,
                p=0.4
            ),
            
            # Brightness and contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4
            ),
            
            # Noise simulation
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 30.0)),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            ], p=0.3),
            
            # Blur effects (simulate motion/acquisition artifacts)
            A.OneOf([
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.2),
            
            # Elastic transformation (very mild for ECG)
            A.ElasticTransform(
                alpha=30,
                sigma=5,
                alpha_affine=3,
                p=0.1
            ),
            
            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            )
        ])
    
    def _create_aggressive_pipeline(self):
        """Create aggressive augmentation pipeline for data-scarce scenarios"""
        return A.Compose([
            # More aggressive geometric changes
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.15,
                rotate_limit=8,
                p=0.5
            ),
            
            # Color space adjustments
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5
            ),
            
            # Multiple noise types
            A.OneOf([
                A.GaussNoise(var_limit=(15.0, 50.0)),
                A.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 0.8)),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
            ], p=0.4),
            
            # Distortions
            A.OneOf([
                A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5),
                A.GridDistortion(num_steps=3, distort_limit=0.15),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05),
            ], p=0.2),
            
            # Blur and sharpening
            A.OneOf([
                A.GaussianBlur(blur_limit=5),
                A.MotionBlur(blur_limit=5),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
            ], p=0.3),
            
            # Coarse dropout (simulate electrode disconnection)
            A.CoarseDropout(
                max_holes=3,
                max_height=20,
                max_width=20,
                min_holes=1,
                min_height=5,
                min_width=5,
                p=0.1
            ),
            
            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            )
        ])
    
    def _create_clinical_pipeline(self):
        """Create clinically-inspired augmentation pipeline"""
        return A.Compose([
            # Simulate clinical acquisition variations
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4,
                brightness_by_max=False
            ),
            
            # Simulate different ECG machine characteristics
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 25.0)),  # Electrical noise
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3)),  # Sensor noise
            ], p=0.3),
            
            # Simulate paper/printing variations
            A.OneOf([
                A.RandomGamma(gamma_limit=(80, 120)),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
            ], p=0.2),
            
            # Simulate patient movement (very minimal)
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.05,
                rotate_limit=3,
                p=0.3
            ),
            
            # Simulate lead placement variations
            A.ElasticTransform(
                alpha=20,
                sigma=3,
                alpha_affine=2,
                p=0.1
            ),
            
            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            )
        ])
    
    def augment_image(self, image, pipeline_type='moderate', return_original=False):
        """
        Apply augmentation to a single image
        
        Args:
            image (np.ndarray): Input image
            pipeline_type (str): Augmentation pipeline ('mild', 'moderate', 'aggressive', 'clinical')
            return_original (bool): Whether to return original image as well
            
        Returns:
            np.ndarray or tuple: Augmented image(s)
        """
        # Select pipeline
        pipelines = {
            'mild': self.mild_pipeline,
            'moderate': self.moderate_pipeline,
            'aggressive': self.aggressive_pipeline,
            'clinical': self.clinical_pipeline
        }
        
        pipeline = pipelines.get(pipeline_type, self.moderate_pipeline)
        
        # Apply augmentation
        augmented = pipeline(image=image)
        augmented_image = augmented['image']
        
        if return_original:
            return image, augmented_image
        return augmented_image
    
    def create_tf_dataset(self, directory, batch_size=32, target_size=None, 
                         augmentation_type='moderate', validation_split=0.0):
        """
        Create TensorFlow dataset with CNN-optimized augmentations
        
        Args:
            directory (str): Path to image directory
            batch_size (int): Batch size
            target_size (tuple): Target image size
            augmentation_type (str): Type of augmentation to apply
            validation_split (float): Fraction for validation split
            
        Returns:
            tf.data.Dataset: Configured TensorFlow dataset
        """
        if target_size is None:
            target_size = self.target_size
        
        # Create dataset from directory
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            validation_split=validation_split,
            subset="training" if validation_split > 0 else None,
            seed=42,
            image_size=target_size,
            batch_size=batch_size
        )
        
        # Apply preprocessing and augmentation
        def preprocess_and_augment(image, label):
            # Convert to float32 and normalize
            image = tf.cast(image, tf.float32) / 255.0
            
            if augmentation_type != 'none':
                # Apply random augmentations based on type
                if augmentation_type in ['moderate', 'aggressive']:
                    # Random rotation
                    rotation_factor = 0.05 if augmentation_type == 'moderate' else 0.1
                    image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
                    
                    # Random brightness and contrast
                    brightness_delta = 0.1 if augmentation_type == 'moderate' else 0.2
                    contrast_range = [0.9, 1.1] if augmentation_type == 'moderate' else [0.8, 1.2]
                    
                    image = tf.image.random_brightness(image, brightness_delta)
                    image = tf.image.random_contrast(image, contrast_range[0], contrast_range[1])
                
                # Clip values to valid range
                image = tf.clip_by_value(image, 0.0, 1.0)
            
            return image, label
        
        # Apply preprocessing
        dataset = dataset.map(preprocess_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Configure for performance
        dataset = dataset.cache()
        dataset = dataset.shuffle(1000)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def augment_dataset(self, input_dir, output_dir, augmentations_per_image=5,
                       pipeline_type='moderate', classes=None):
        """
        Augment entire dataset and save to new directory
        
        Args:
            input_dir (str): Input dataset directory
            output_dir (str): Output directory for augmented images
            augmentations_per_image (int): Number of augmentations per original image
            pipeline_type (str): Augmentation pipeline type
            classes (list): Specific classes to augment (None for all)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        print(f"🔄 Augmenting dataset: {input_dir} → {output_dir}")
        print(f"📊 Augmentations per image: {augmentations_per_image}")
        print(f"🎨 Pipeline type: {pipeline_type}")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get class folders
        class_folders = [f for f in input_path.iterdir() if f.is_dir()]
        if classes:
            class_folders = [f for f in class_folders if f.name in classes]
        
        total_original = 0
        total_augmented = 0
        
        for class_folder in class_folders:
            class_name = class_folder.name
            class_output_dir = output_path / class_name
            class_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n📁 Processing class: {class_name}")
            
            # Get image files
            image_files = list(class_folder.glob('*.png'))
            total_original += len(image_files)
            
            print(f"   Original images: {len(image_files)}")
            
            for img_file in image_files:
                try:
                    # Load original image
                    image = cv2.imread(str(img_file))
                    if image is None:
                        continue
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, self.target_size)
                    
                    # Copy original image
                    original_name = f"{img_file.stem}_original.png"
                    original_output = class_output_dir / original_name
                    Image.fromarray(image).save(original_output)
                    
                    # Generate augmentations
                    for aug_idx in range(augmentations_per_image):
                        augmented = self.augment_image(image, pipeline_type)
                        
                        # Convert back to uint8 for saving
                        if augmented.max() <= 1.0:
                            augmented = (augmented * 255).astype(np.uint8)
                        
                        # Save augmented image
                        aug_name = f"{img_file.stem}_aug_{aug_idx:02d}.png"
                        aug_output = class_output_dir / aug_name
                        Image.fromarray(augmented).save(aug_output)
                        
                        total_augmented += 1
                
                except Exception as e:
                    print(f"❌ Error processing {img_file}: {e}")
            
            class_total = len(list(class_output_dir.glob('*.png')))
            print(f"   Total images (original + augmented): {class_total}")
        
        print(f"\n✅ Dataset augmentation completed!")
        print(f"📊 Original images: {total_original}")
        print(f"🎨 Augmented images: {total_augmented}")
        print(f"📁 Total images: {total_original + total_augmented}")
        print(f"💾 Saved to: {output_dir}")
    
    def visualize_augmentations(self, image_path, pipeline_type='moderate', 
                               num_augmentations=8, save_path=None):
        """
        Visualize different augmentations of a single image
        
        Args:
            image_path (str): Path to input image
            pipeline_type (str): Augmentation pipeline type
            num_augmentations (int): Number of augmentations to show
            save_path (str): Path to save visualization
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)
        
        # Create subplot grid
        cols = 4
        rows = (num_augmentations + 3) // 4  # +3 for original + ceil division
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten() if rows * cols > 1 else [axes]
        
        # Show original
        axes[0].imshow(image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Generate and show augmentations
        for i in range(1, min(num_augmentations + 1, len(axes))):
            augmented = self.augment_image(image, pipeline_type)
            
            # Handle normalized images
            if augmented.max() <= 1.0:
                # Denormalize for visualization
                if len(augmented.shape) == 3 and augmented.shape[2] == 3:
                    # RGB denormalization
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    augmented = augmented * std + mean
                    augmented = np.clip(augmented, 0, 1)
            
            axes[i].imshow(augmented)
            axes[i].set_title(f'Augmented {i}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_augmentations + 1, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'ECG Augmentations - {pipeline_type.title()} Pipeline', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Augmentation visualization saved: {save_path}")
        
        plt.show()
    
    def compare_pipelines(self, image_path, save_path=None):
        """
        Compare different augmentation pipelines on the same image
        
        Args:
            image_path (str): Path to input image
            save_path (str): Path to save comparison
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)
        
        # Pipeline types
        pipelines = ['mild', 'moderate', 'aggressive', 'clinical']
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Show original
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Apply each pipeline
        for i, pipeline_type in enumerate(pipelines):
            augmented = self.augment_image(image, pipeline_type)
            
            # Handle normalized images for visualization
            if augmented.max() <= 1.0:
                if len(augmented.shape) == 3 and augmented.shape[2] == 3:
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    augmented = augmented * std + mean
                    augmented = np.clip(augmented, 0, 1)
            
            axes[i + 1].imshow(augmented)
            axes[i + 1].set_title(f'{pipeline_type.title()} Pipeline')
            axes[i + 1].axis('off')
        
        # Hide last subplot
        axes[5].axis('off')
        
        plt.suptitle('ECG Augmentation Pipeline Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Pipeline comparison saved: {save_path}")
        
        plt.show()


def main():
    """Main function for testing augmentation"""
    augmentor = ECGAugmentor()
    
    # Find a sample image
    sample_dirs = ['train', 'test']
    sample_image = None
    
    for dir_name in sample_dirs:
        if os.path.exists(dir_name):
            for class_dir in os.listdir(dir_name):
                class_path = os.path.join(dir_name, class_dir)
                if os.path.isdir(class_path):
                    images = [f for f in os.listdir(class_path) if f.endswith('.png')]
                    if images:
                        sample_image = os.path.join(class_path, images[0])
                        break
            if sample_image:
                break
    
    if sample_image:
        print(f"🖼️ Using sample image: {sample_image}")
        
        # Visualize augmentations
        augmentor.visualize_augmentations(
            sample_image, 
            pipeline_type='moderate',
            save_path='ecg_augmentation_samples.png'
        )
        
        # Compare pipelines
        augmentor.compare_pipelines(
            sample_image,
            save_path='ecg_pipeline_comparison.png'
        )
        
    else:
        print("❌ No sample images found in train/ or test/ directories")
        print("💡 Place ECG images in the appropriate directory structure to test augmentation")


if __name__ == "__main__":
    main()