import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load trained model
model = load_model("model/emotion_model.h5")

# Emotion labels
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Path to folder containing test images
folder_path = r"face_detector\validation\Angry"  # You can change this to any class folder

# Loop through all images in folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"❌ Skipped {filename} - could not read image")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (48, 48))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face)
        label = classes[np.argmax(prediction)]

        print(f"✅ {filename}: {label}")
