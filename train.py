#!/usr/bin/env python3
"""
train.py - Train a deep learning model for ASL dynamic gesture recognition
"""

import os
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_DIR = 'models'
FRAMES_PER_VIDEO = 16
FRAME_HEIGHT = 64
FRAME_WIDTH = 64
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2
RANDOM_STATE = 42

def create_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(MODEL_DIR, exist_ok=True)

def load_video_frames(video_path, num_frames=FRAMES_PER_VIDEO):
    """
    Load and preprocess frames from a video file
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        
    Returns:
        Numpy array of shape (num_frames, height, width, channels)
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    # Calculate frame indices to sample uniformly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Resize and normalize
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        else:
            # If frame reading fails, duplicate the last valid frame
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32))
    
    cap.release()
    
    # Ensure we have exactly num_frames frames
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32))
    
    return np.array(frames[:num_frames])

def load_dataset(data_dir):
    """
    Load all videos from the dataset directory
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        X: Numpy array of videos (n_samples, n_frames, height, width, channels)
        y: Numpy array of labels
        class_names: List of class names
    """
    X = []
    y = []
    class_names = []
    
    # Get all class directories
    class_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir()]
    class_dirs.sort()
    
    print(f"Found {len(class_dirs)} classes")
    
    for class_idx, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        class_names.append(class_name)
        
        # Get all video files in this class directory
        video_files = list(class_dir.glob('*.avi')) + list(class_dir.glob('*.mp4'))
        
        print(f"Loading {len(video_files)} videos from class '{class_name}'...")
        
        for video_file in tqdm(video_files, desc=f"Class {class_name}"):
            frames = load_video_frames(video_file)
            
            if frames is not None:
                X.append(frames)
                y.append(class_idx)
    
    return np.array(X), np.array(y), class_names

def create_cnn_lstm_model(input_shape, num_classes):
    """
    Create a CNN + LSTM model for video classification
    
    Args:
        input_shape: Shape of input videos (frames, height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # TimeDistributed wrapper applies the same CNN to each frame
        layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', padding='same'), 
                             input_shape=input_shape),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.BatchNormalization()),
        
        layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.BatchNormalization()),
        
        layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.BatchNormalization()),
        
        # Flatten spatial dimensions for each frame
        layers.TimeDistributed(layers.Flatten()),
        
        # LSTM layers to process temporal information
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.5),
        layers.LSTM(64),
        layers.Dropout(0.5),
        
        # Dense layers for classification
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Main training function"""
    print("ASL Dynamic Gesture Recognition Training")
    print("=" * 50)
    
    # Create necessary directories
    create_directories()
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found!")
        print("Please create the directory and add your video data.")
        return
    
    # Load dataset
    print("\n1. Loading dataset...")
    X, y, class_names = load_dataset(DATA_DIR)
    
    if len(X) == 0:
        print("Error: No videos found in the dataset!")
        return
    
    print(f"\nDataset loaded successfully!")
    print(f"Total samples: {len(X)}")
    print(f"Video shape: {X.shape}")
    print(f"Classes: {class_names}")
    
    # Create label encoder and save it
    le = LabelEncoder()
    le.fit(range(len(class_names)))
    le.classes_ = np.array(class_names)
    joblib.dump(le, os.path.join(MODEL_DIR, 'le.pkl'))
    print(f"\nLabel encoder saved to {os.path.join(MODEL_DIR, 'le.pkl')}")
    
    # Convert labels to categorical
    y_categorical = to_categorical(y, num_classes=len(class_names))
    
    # Split dataset
    print("\n2. Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=TEST_SPLIT, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create model
    print("\n3. Creating model...")
    input_shape = (FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, 3)
    model = create_cnn_lstm_model(input_shape, len(class_names))
    
    print("\nModel architecture:")
    model.summary()
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, 'asl_model_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\n4. Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model_path = os.path.join(MODEL_DIR, 'asl_model.h5')
    model.save(model_path)
    print(f"\n5. Model saved to {model_path}")
    
    # Evaluate model
    print("\n6. Model evaluation:")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTraining accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {test_acc:.4f}")
    
    # Save training history
    history_data = {
        'train_acc': history.history['accuracy'],
        'val_acc': history.history['val_accuracy'],
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }
    np.save(os.path.join(MODEL_DIR, 'training_history.npy'), history_data)
    
    print("\nTraining completed successfully!")
    print(f"\nSaved files:")
    print(f"- Model: {os.path.join(MODEL_DIR, 'asl_model.h5')}")
    print(f"- Best model: {os.path.join(MODEL_DIR, 'asl_model_best.h5')}")
    print(f"- Label encoder: {os.path.join(MODEL_DIR, 'le.pkl')}")
    print(f"- Training history: {os.path.join(MODEL_DIR, 'training_history.npy')}")

if __name__ == "__main__":
    main()