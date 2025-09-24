#!/usr/bin/env python3
"""
🫀 RhythmGuard CNN Model
========================
Convolutional Neural Network implementation for ECG rhythm classification.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

class RhythmGuardCNN:
    """
    CNN Model for ECG Rhythm Classification
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=6, model_name="rythmguard_cnn"):
        """
        Initialize the CNN model
        
        Args:
            input_shape (tuple): Input image shape (height, width, channels)
            num_classes (int): Number of classification classes
            model_name (str): Name for the model
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = None
        self.history = None
        self.class_names = ['F', 'M', 'N', 'Q', 'S', 'V']  # ECG rhythm classes
        
    def build_model(self, architecture='custom'):
        """
        Build the CNN architecture
        
        Args:
            architecture (str): Architecture type ('custom', 'vgg16', 'resnet50')
        """
        if architecture == 'custom':
            self.model = self._build_custom_cnn()
        elif architecture == 'vgg16':
            self.model = self._build_vgg16_transfer()
        elif architecture == 'resnet50':
            self.model = self._build_resnet50_transfer()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
            
        print(f"✅ Built {architecture} model with {self.model.count_params():,} parameters")
        return self.model
    
    def _build_custom_cnn(self):
        """Build a custom CNN architecture optimized for ECG images"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling instead of Flatten to reduce parameters
            layers.GlobalAveragePooling2D(),
            
            # Dense Layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(self.num_classes, activation='softmax', name='predictions')
        ])
        
        return model
    
    def _build_vgg16_transfer(self):
        """Build VGG16-based transfer learning model"""
        base_model = keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _build_resnet50_transfer(self):
        """Build ResNet50-based transfer learning model"""
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """Compile the model with optimizer and loss function"""
        if optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizer
            
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ Model compiled with {optimizer} optimizer (lr={learning_rate})")
    
    def create_data_generators(self, train_dir, test_dir=None, validation_split=0.2, 
                             batch_size=32, augment_data=True):
        """
        Create data generators for training and validation
        
        Args:
            train_dir (str): Path to training data directory
            test_dir (str): Path to test data directory (optional)
            validation_split (float): Fraction of training data to use for validation
            batch_size (int): Batch size for training
            augment_data (bool): Whether to apply data augmentation
        """
        if augment_data:
            # Data augmentation for training
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=False,  # ECG signals shouldn't be flipped
                validation_split=validation_split
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split
            )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Test generator (if test directory provided)
        test_generator = None
        if test_dir and os.path.exists(test_dir):
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=self.input_shape[:2],
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            )
        
        print(f"✅ Data generators created:")
        print(f"   📁 Training samples: {train_generator.samples}")
        print(f"   📁 Validation samples: {validation_generator.samples}")
        if test_generator:
            print(f"   📁 Test samples: {test_generator.samples}")
        print(f"   🎯 Classes: {list(train_generator.class_indices.keys())}")
        
        self.class_names = list(train_generator.class_indices.keys())
        
        return train_generator, validation_generator, test_generator
    
    def train_model(self, train_generator, validation_generator, epochs=50, 
                   callbacks_list=None, verbose=1):
        """
        Train the CNN model
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs (int): Number of training epochs
            callbacks_list (list): List of Keras callbacks
            verbose (int): Verbosity level
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if callbacks_list is None:
            callbacks_list = self._get_default_callbacks()
        
        print(f"🚀 Starting training for {epochs} epochs...")
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        print("✅ Training completed!")
        return self.history
    
    def _get_default_callbacks(self):
        """Get default callbacks for training"""
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                f'{self.model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks_list
    
    def evaluate_model(self, test_generator):
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print("📊 Evaluating model on test data...")
        
        # Get predictions
        test_generator.reset()
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = test_generator.classes
        
        # Calculate accuracy
        accuracy = np.mean(predicted_classes == true_classes)
        
        print(f"🎯 Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=self.class_names,
            output_dict=True
        )
        
        print("\n📈 Classification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('CNN Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, report, cm
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("❌ No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        if filepath is None:
            filepath = f'{self.model_name}.h5'
        
        self.model.save(filepath)
        
        # Also save model metadata
        metadata = {
            'model_name': self.model_name,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'architecture': 'CNN',
            'framework': 'TensorFlow/Keras'
        }
        
        joblib.dump(metadata, filepath.replace('.h5', '_metadata.joblib'))
        
        print(f"✅ Model saved: {filepath}")
        print(f"✅ Metadata saved: {filepath.replace('.h5', '_metadata.joblib')}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        
        # Load metadata if available
        metadata_path = filepath.replace('.h5', '_metadata.joblib')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.class_names = metadata.get('class_names', self.class_names)
            self.input_shape = metadata.get('input_shape', self.input_shape)
            self.num_classes = metadata.get('num_classes', self.num_classes)
        
        print(f"✅ Model loaded: {filepath}")
    
    def predict_single_image(self, image_path):
        """Predict a single ECG image"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(image_path, target_size=self.input_shape[:2])
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        result = {
            'predicted_class': self.class_names[predicted_class],
            'confidence': float(confidence),
            'all_probabilities': {self.class_names[i]: float(predictions[0][i]) 
                                for i in range(len(self.class_names))}
        }
        
        return result