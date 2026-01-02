# AI-Based Facial Emotion Detection System

This project is a real-time facial emotion detection system that uses a Convolutional Neural Network (CNN) to classify human emotions from live webcam input. The system detects faces using Haar Cascade and predicts emotions such as Angry, Happy, Sad, Neutral, Surprise, Fear, and Disgust.

## 🚀 Features

- Real-time face detection using Haar Cascade
- Emotion classification using a trained CNN model
- Supports 7 emotion classes
- Live webcam-based emotion recognition
- Mirrored camera view for natural interaction
- Lightweight and runs locally

## 📸 Screenshots

> Below are screenshots of the working facial emotion detection model running on a local machine.

<img width="1920" height="1080" alt="Screenshot 2026-01-02 190233" src="https://github.com/user-attachments/assets/2bc9936d-1790-4cc2-a60c-4ae42571b325" />


- Live webcam emotion detection
- Face bounding box with emotion label
- Real-time prediction output
  
## 🛠️ Tech Stack

- Python
- TensorFlow & Keras
- OpenCV (cv2)
- NumPy
- Haar Cascade Classifier
- CNN (Convolutional Neural Network)
- Kaggle Facial Emotion Dataset

## 📊 Dataset

The model was trained using a Facial Emotion Recognition dataset sourced from Kaggle.

- Image size: 48x48 pixels
- Color format: Grayscale
- Total classes: 7 emotions
- Dataset includes labeled facial expressions for supervised learning

## ⚙️ How It Works

1. Webcam captures live video frames
2. Frames are mirrored for natural viewing
3. Converted to grayscale
4. Faces are detected using Haar Cascade
5. Face region is resized and normalized
6. CNN model predicts the emotion
7. Emotion label is displayed on screen


## ⚡ Installation & Usage

### Prerequisites
- Python 3.9 – 3.10 recommended
- Webcam

### Install dependencies
```bash
pip install numpy opencv-python tensorflow

python detection.py

## ⚠️ Limitations

- Performance depends on lighting conditions
- Accuracy decreases for side-face or occluded faces
- Emotion predictions may fluctuate frame-to-frame
- Model accuracy depends on dataset quality

## 🔮 Future Enhancements

- Improve accuracy using larger datasets
- Add confidence score for predictions
- Deploy as a web or mobile application
- Implement emotion smoothing across frames
- Add age and gender detection

## 👤 Author

Developed by **Smitesh Pokharkar**  
B.Tech CSE | AI & Machine Learning Enthusiast
