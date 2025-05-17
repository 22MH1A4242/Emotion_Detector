# 😄 Emotion Detector

A deep learning-based emotion detection web application built using TensorFlow, Keras, OpenCV, and Flask. It can detect emotions from facial expressions in real-time or from images.

## 📂 Project Structure

emotion-detector/
├── app.py # Main Flask web app
├── webcam_predict.py # Real-time emotion detection from webcam
├── train_model.py # Model training script
├── test_predict.py # Prediction on test images
├── batch_predict.py # Batch image prediction
├── prepare_dataset.py # Dataset preprocessing
├── model/
│ └── emotion_model.h5 # Trained model
├── face_detector/
│ ├── train/ # Training images
│ └── validation/ # Validation images
├── static/ # Static files (CSS, JS)
├── templates/ # HTML templates
└── .gitignore # Ignored files (e.g., myenv)


## ⚙️ Features

- Detects 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- Trained CNN model using FER-2013 dataset format
- Real-time webcam emotion detection
- Batch and single image testing
- Flask-based web interface

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/22MH1A4242/emotion-detector.git
   cd emotion-detector
Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
Run the app:
python app.py

🧠 Model Training
To train your own model:
python train_model.py
Ensure your face_detector/train and face_detector/validation directories are properly structured.

📸 Webcam Detection
python webcam_predict.py

🖼️ Test with Images
python test_predict.py

.gitignore
This prevents unnecessary files and folders (like virtual environments, cache, and system files) from being tracked by Git.
# Virtual Environment
myenv/
venv/
env/
# Python cache
__pycache__/
*.py[cod]
# Jupyter Notebook checkpoints
.ipynb_checkpoints/
# System files
.DS_Store
Thumbs.db
# Model files
*.h5
*.keras
# VSCode settings
.vscode/
# Log files
*.log
# TensorBoard logs
logs/
To add it to your repo:

 
requirements.txt
These are the essential Python packages for your project. Adjust versions if needed:
flask==2.3.3
tensorflow==2.19.0
keras==2.14.0
opencv-python==4.9.0.80
matplotlib==3.8.4
numpy==1.26.4
Pillow==10.3.0

👤 Author
Anjali Devi Medapati
CSE (AI & ML), 3rd Year
GitHub: 22MH1A4242

