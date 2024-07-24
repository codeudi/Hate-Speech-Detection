# Hate Speech Detection

## Overview

This project aims to detect hate speech and offensive language using machine learning techniques. It includes capabilities for text, audio, and video inputs to classify content as either non-offensive, offensive, or hate speech.

## Features

- **Text Classification**: Utilizes machine learning models to analyze text inputs.
- **Audio Analysis**: Processes audio files to identify hate speech.
- **Video Content Analysis**: Detects hate speech in video content.

## Technologies Used

- Python
- Machine Learning (Scikit-learn, XGBoost, etc.)
- Streamlit (for UI)
- Pydub, SpeechRecognition (for audio processing)
- MoviePy (for video processing)

## Setup Instructions

1. **Clone the repository:**
   git clone <repository url>

2. **Install the dependencies**
    pip install -r requirements.txt

3. **Run the Application**
    streamlit run app.py

## Usage 

- Text Input: Enter text to classify whether it contains hate speech.
- Audio Input: Upload an audio file to analyze speech content.
- Video Input: Upload a video file to detect hate speech in its audio track.


