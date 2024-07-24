import streamlit as st
import re
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import joblib
import os
import pickle

# Load model and vectorizer
model = joblib.load("models/ensemble_classifier.joblib")
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define labels
class_labels = {
    0: 'Hate Speech',
    1: 'Offensive Language',
    2: 'Not Offensive Language'
}

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# Function to classify text
def classify_text(text):
    preprocessed_text = preprocess_text(text)
    tfidf_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(tfidf_text)
    return class_labels.get(prediction[0], 'Unknown')

# Function to convert audio to text
def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    sound = AudioSegment.from_file(audio_path, format="wav")
    chunks = split_on_silence(sound, min_silence_len=700, silence_thresh=sound.dBFS-16, keep_silence=500)
    text = ""
    for i, chunk in enumerate(chunks):
        chunk.export(f"chunk{i}.wav", format="wav")
        with sr.AudioFile(f"chunk{i}.wav") as source:
            audio_listened = recognizer.record(source)
            try:
                text_chunk = recognizer.recognize_google(audio_listened)
                text += " " + text_chunk.strip()  # Ensure words are properly concatenated
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                st.error(f"Could not request results; {e}")
    return text.strip()

# Function to convert video to audio and then extract text
def video_to_text(video_path):
    audio_path = 'temp_audio.wav'
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path)
    return audio_to_text(audio_path)


# Set page configuration and custom CSS
st.set_page_config(page_title='Hate Speech Detection', page_icon='🗣️', layout='wide')

# Define custom CSS for styling
custom_css = """
<style>
    body {
        background-color: #f0f0f0;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #FFF6E9;
    }
    [data-testid=stSidebar]{
        background-color:black;
        
    }
    .main-content {
        padding: 20px;
    }
    .section {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .section h2 {
        font-size: 24px;
        font-weight: bold;
        color: #405d72;
        margin-bottom: 10px;
    }
    .section p {
        font-size: 16px;
        color: #758694;
        line-height: 1.6;
    }
    .prediction-box {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .not-offensive {
        background-color: #c8e6c9;
        color: #2e7d32;
    }
    .offensive {
        background-color: #ffcdd2;
        color: #b71c1c;
    }
    .landing-page {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        text-align: center;
        border: 2px solid #FF6969;
        padding:40px;
        border-radius:10px;
    }
    [data-testid=stHeadingWithActionElements]{
        width:100%;
        height:30%;
    }
    [data-testid=baseButton-secondary]{
        background-color: #FFF8F3;
        color:black;
    }
    .landing-page h1 {
        font-size:60px;
        color: #C80036;
    }
    .landing-page p{
        color: #0C1844;
        font-weight:bold;
    }
    .landing-page img{
        margin-bottom:20px;
    }
    .landing-page .button-container {
        display: flex;
        justify-content: space-around;
        width: 50%;
        margin-top: 20px;
    }
    .landing-page .button-container .btn {
        padding: 10px 20px;
        background-color: #405d72;
        color: #ffffff;
        border: none;
        border-radius: 5px;
        text-decoration: none;
        font-size: 18px;
        transition: all 0.3s ease;
    }
    .stTextArea textarea {
        background-color: white;
        color: black;
        font-size: 28px;
        caret-color: black;
    }
    .st-emotion-cache-qgowjl p {
        font-size: 20px;
        color: #FF6969;
        font-weight:bold;
    }
    .st-emotion-cache-fm8pe0 p{
        font-weight:bold;
    }
    .st-emotion-cache-1sno8jx p{
        font-size: 20px;
        color: #0C1844;
        font-weight:bold;
    }
    [data-testid=baseButton-secondary] {
        background-color: #C80036;
        color: white;
    }
    .st-emotion-cache-1uixxvy{
        color:#0C1844;
        font-weight:bold;
    }
    
    .st-emotion-cache-4mjat2{
        color:#0C1844;
    }

    .stAlert{
        background-color: red;
        border-radius : 10px;
    }
    .coloring{
        color:#FF6969;
        font-weight:bold;
    }
    .nav{
        display:flex;
        gap:20px;
    }
    .nav img{
        margin-top:13px;
    }
    .tinp{
        display:flex;
    }
    .tinp img{
        margin-top:8px;
    }
    .ainp{
        display:flex;
    }
    .ainp img{
        margin-top:8px;
    }
    .vinp{
        display:flex;
    }
    .vinp img{
        margin-top:8px;
    }
</style>
"""

# Render custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("<div class='nav'><h1 style='color: white;'>Navigation Menu</h1> <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS1vlSPY8OwPQ1_EwUsPOHl873Mn7iTqVZyPA&s' height='40px' width='40px'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='border: 1px solid white;'>", unsafe_allow_html=True)

# Initialize session state
if 'navigation' not in st.session_state:
    st.session_state.navigation = "Home"

# Check for button clicks and update navigation state
if st.sidebar.button("Home"):
    st.session_state.navigation = "Home"

if st.sidebar.button("Text Input"):
    st.session_state.navigation = "Text Input"

if st.sidebar.button("Audio Input"):
    st.session_state.navigation = "Audio Input"

if st.sidebar.button("Video Input"):
    st.session_state.navigation = "Video Input"

st.sidebar.markdown("<hr style='border: 1px solid white;'>", unsafe_allow_html=True)

# Main content based on navigation
if st.session_state.navigation == "Home":
    st.markdown("""
    <div class="landing-page" style="text-align:center;">
        <h1>Hate Speech Detection</h1>
        <img src='https://www.un.org/sites/un2.un.org/files/tracking_hate_speech.png' height='100px' width='100px'>
        <p><span class="coloring">Hate speech detection</span> involves using technology to identify offensive language, such as derogatory remarks, threats, and discriminatory statements in <span class="coloring">text, audio, and video</span>. Our app uses <span class="coloring">machine learning</span> and NLP to swiftly flag such content, maintaining user safety and community standards.

The app employs <span class="coloring">deep learning</span> to categorize content as <span class="coloring">"Hate Speech," "Offensive Language," or "Not Offensive Language,"</span> enabling real-time moderation. This helps platforms manage content effectively and promote healthy online interactions.

Adaptable to new expressions, the app enhances accuracy in detecting nuanced offensive language, ensuring effective identification and mitigation of emerging threats, and supporting respectful online discourse.</p>
    </div>
    """, unsafe_allow_html=True)


    #st.markdown('<h1 style="text-align:center; border: 2px solid white;margin-bottom:10px; padding-left:3px;">Get Started with Hate Speech Detection</h1>',unsafe_allow_html=True)
elif st.session_state.navigation == "Text Input":
    st.markdown('<div class="tinp"><h1 style="color: #C80036">Text Input</h1> <img src="https://cdn-icons-png.flaticon.com/512/7039/7039832.png" height="50px" width="50px"> </div>', unsafe_allow_html=True)
    text_input = st.text_area('Enter text to classify:')
    if st.button('Classify Text'):
        if text_input.strip():
            prediction = classify_text(text_input)
            if prediction == 'Not Offensive Language':
                st.markdown(f'<div class="prediction-box not-offensive">{prediction}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box offensive">{prediction}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to classify.")

    # Audio Input Section
elif st.session_state.navigation == "Audio Input":
    st.markdown('<div class="ainp"><h1 style="color: #C80036">Audio Input</h1> <img src="https://cdn-icons-png.flaticon.com/512/5525/5525198.png" height="70px" width="70px"></div>', unsafe_allow_html=True)
    audio_file = st.file_uploader('Upload an audio file:', type=['wav'])
    if audio_file is not None:
        audio_path = f'uploaded_audio/{audio_file.name}'
        with open(audio_path, 'wb') as f:
            f.write(audio_file.getbuffer())
        st.audio(audio_file, format='audio/wav')  # Display audio player
        st.write("Processing audio file... (This may take a moment)")
        audio_text = audio_to_text(audio_path)
        st.write(f'Extracted Text from Audio: {audio_text}')
        if audio_text:
            audio_prediction = classify_text(audio_text)
            if audio_prediction == 'Not Offensive Language':
                st.markdown(f'<div class="prediction-box not-offensive">{audio_prediction}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box offensive">{audio_prediction}</div>', unsafe_allow_html=True)
        else:
            st.warning("No audio content detected. Please upload an audio file with clear speech.")
        os.remove(audio_path)  # Remove temporary audio file

    # Video Input Section
elif st.session_state.navigation == "Video Input":
    st.markdown('<div class="vinp"><h1 style="color: #C80036">Video Input</h1> <img src="https://cdn-icons-png.flaticon.com/512/1950/1950093.png"  height="60px" width="60px"></div>', unsafe_allow_html=True)
    video_file = st.file_uploader('Upload a video file:', type=['mp4', 'avi', 'mov'])
    if video_file is not None:
        video_path = f'uploaded_videos/{video_file.name}'
        with open(video_path, 'wb') as f:
            f.write(video_file.getbuffer())
        st.video(video_file)  # Display video player
        st.write("Processing video file... (This may take a moment)")
        video_text = video_to_text(video_path)
        st.write(f'Extracted Text from Video: {video_text}')
        if video_text:
            video_prediction = classify_text(video_text)
            if video_prediction == 'Not Offensive Language':
                st.markdown(f'<div class="prediction-box not-offensive">{video_prediction}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box offensive">{video_prediction}</div>', unsafe_allow_html=True)
        else:
            st.warning("No video content detected. Please upload a video file with clear speech.")
        os.remove(video_path)  # Remove temporary video file
