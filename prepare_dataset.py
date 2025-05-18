import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('fer2013.csv')

# Emotion labels
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Create folders
base_dir = 'face_detector'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

for dir_path in [train_dir, val_dir]:
    os.makedirs(dir_path, exist_ok=True)
    for emotion in emotion_labels.values():
        os.makedirs(os.path.join(dir_path, emotion), exist_ok=True)

# Split data
train_data, val_data = train_test_split(data, test_size=0.2, stratify=data['emotion'], random_state=42)

def save_images(dataset, folder):
    for i, row in dataset.iterrows():
        emotion = emotion_labels[row['emotion']]
        pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8).reshape(48, 48)
        img = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
        path = os.path.join(base_dir, folder, emotion, f"{i}.jpg")
        cv2.imwrite(path, img)

# Save images to folders
save_images(train_data, 'train')
save_images(val_data, 'validation')

print("Images saved to face_detector/train and face_detector/validation")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # If not already
)
