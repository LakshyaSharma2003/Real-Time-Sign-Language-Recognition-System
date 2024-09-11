Real-Time Indian Sign Language Recognition System
Overview
This project is designed to recognize Indian Sign Language (ISL) gestures in real-time using a webcam. The system utilizes machine learning models for hand gesture recognition and provides real-time translation of gestures into English, Hindi, and Tamil. Additionally, it features text-to-speech functionality to vocalize the recognized gestures.

Key Features
Real-Time Gesture Recognition: Detects and recognizes ISL gestures using a webcam.
Multilingual Translation: Translates recognized gestures into English, Hindi, and Tamil.
Text-to-Speech: Converts recognized gestures into speech in English.
User Interface: Displays recognized gestures and translations directly on the video feed.
Performance: Utilizes GPU acceleration for efficient processing and high frame rates.
Technologies Used
Python: Programming language for implementing the project.
OpenCV: For real-time computer vision and hand gesture detection.
cvzone: Hand tracking and gesture classification module.
TensorFlow/Keras: For gesture classification using a pre-trained model.
Google Translate API: For translating text into Hindi and Tamil.
pyttsx3: Text-to-speech conversion.
Pillow: For rendering multilingual text on images.
CUDA & cuDNN: GPU acceleration for improved performance.
Setup Instructions
Install Dependencies:

Ensure Python is installed.
Install required Python packages:
bash
Copy code
pip install opencv-python cvzone tensorflow googletrans==4.0.0-rc1 pyttsx3 pillow
Install CUDA and cuDNN:

Download and install CUDA 11.8 and cuDNN 8.6 compatible with TensorFlow 2.12.0.
Set the environment variables for CUDA and cuDNN.
Download the Model:

Place the pre-trained gesture classification model (keras_model.h5) and labels file (labels.txt) in the appropriate directory.
Run the Application:

Execute the Python script to start the real-time sign language recognition system:
bash
Copy code
python your_script_name.py
Contribution
Feel free to contribute by submitting issues, suggesting features, or making pull requests. For any queries or issues, please open an issue in this repository.


