import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('model/emotion_model.h5')

# Load and preprocess the image
image_path = r'face_detector\validation\Angry\PrivateTest_88305.jpg'  # You can change this path to test other images
image = cv2.imread(image_path)

if image is None:
    print(f"❌ Error: Could not load image at {image_path}")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (48, 48))
normalized = resized.astype('float32') / 255.0
reshaped = np.reshape(normalized, (1, 48, 48, 1))

# Predict
predictions = model.predict(reshaped)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
predicted_emotion = emotion_labels[np.argmax(predictions)]

print(f"✅ Predicted Emotion: {predicted_emotion}")

