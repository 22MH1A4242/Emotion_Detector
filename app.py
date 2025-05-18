from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_emotion = ''
    error_message = ''
    
    if request.method == 'POST':
        file = request.files.get('image')

        if not file or file.filename == '':
            error_message = 'No image uploaded. Please select an image file.'
            return render_template('index.html', error=error_message)

        if not allowed_file(file.filename):
            error_message = 'Invalid file type. Allowed types: png, jpg, jpeg.'
            return render_template('index.html', error=error_message)

        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        # Read and preprocess the image
        image = cv2.imread(filepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized.astype('float32') / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))

        # Predict emotion
        predictions = model.predict(reshaped)
        predicted_emotion = emotion_labels[np.argmax(predictions)]

        return render_template('index.html', filename=file.filename, emotion=predicted_emotion)

    return render_template('index.html')

if __name__ == '__main__':
    # Run app on all network interfaces for mobile device access
    app.run(debug=True)
