import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import os
import tempfile

class SpeechEmotionDetector:
    def __init__(self, model_path="models/speech_model.h5"):
        self.model = tf.keras.models.load_model(model_path)
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
        
    def extract_features(self, audio_path):
        # Load audio file
        y, sr = librosa.load(audio_path, duration=3, offset=0.5)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        # Reshape for model input
        return mfcc_scaled.reshape(1, -1)
    
    def predict(self, audio_path):
        features = self.extract_features(audio_path)
        predictions = self.model.predict(features)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        return self.emotions[predicted_class], confidence

# Set page configuration
st.set_page_config(
    page_title="Speech Emotion Detection",
    page_icon="🎤",
    layout="wide"
)

# App title and description
st.title("Speech Emotion Detection")
st.write("Upload an audio file to detect the emotion in speech.")

# Initialize the model
@st.cache_resource
def load_model():
    try:
        return SpeechEmotionDetector()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

detector = load_model()

# File uploader for audio
st.subheader("Upload Audio")
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "ogg"])

# Audio recorder option
st.subheader("Or Record Audio")
audio_recorder = st.audio_recorder(
    text="Click to record",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    start_prompt="Recording...",
    stop_prompt="Click to stop",
)

# Process uploaded file
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_filename = tmp_file.name
    
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')
    
    # Process the audio
    if st.button("Detect Emotion from Uploaded File"):
        with st.spinner("Analyzing audio..."):
            try:
                emotion, confidence = detector.predict(temp_filename)
                
                # Create columns for results and visualization
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.success(f"Detected Emotion: **{emotion.capitalize()}**")
                    st.info(f"Confidence: {confidence:.2f}")
                
                with col2:
                    # Display emoji based on emotion
                    emoji_map = {
                        "anger": "😠",
                        "disgust": "🤢",
                        "fear": "😨",
                        "happiness": "😊",
                        "neutral": "😐",
                        "sadness": "😢",
                        "surprise": "😲"
                    }
                    st.markdown(f"<h1 style='text-align: center;'>{emoji_map[emotion]}</h1>", 
                                unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error analyzing audio: {e}")
            
            # Clean up temp file
            os.unlink(temp_filename)

# Process recorded audio
if audio_recorder is not None:
    # If audio was recorded
    if len(audio_recorder) > 0:
        # Save the recorded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_recorder)
            temp_filename = tmp_file.name
        
        # Display audio player for the recorded audio
        st.audio(audio_recorder, format='audio/wav')
        
        # Process the audio
        if st.button("Detect Emotion from Recorded Audio"):
            with st.spinner("Analyzing audio..."):
                try:
                    emotion, confidence = detector.predict(temp_filename)
                    
                    # Create columns for results and visualization
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.success(f"Detected Emotion: **{emotion.capitalize()}**")
                        st.info(f"Confidence: {confidence:.2f}")
                    
                    with col2:
                        # Display emoji based on emotion
                        emoji_map = {
                            "anger": "😠",
                            "disgust": "🤢",
                            "fear": "😨",
                            "happiness": "😊",
                            "neutral": "😐",
                            "sadness": "😢",
                            "surprise": "😲"
                        }
                        st.markdown(f"<h1 style='text-align: center;'>{emoji_map[emotion]}</h1>", 
                                    unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error analyzing audio: {e}")
                
                # Clean up temp file
                os.unlink(temp_filename)

# Add information about the app
with st.expander("About this app"):
    st.write("""
    This app detects emotions in speech using a machine learning model. 
    It can identify seven emotions: anger, disgust, fear, happiness, neutral, sadness, and surprise.
    
    To use the app:
    1. Upload an audio file or record using your microphone
    2. Click 'Detect Emotion' to analyze the audio
    3. The app will display the detected emotion and confidence level
    
    Note: For best results, ensure the audio is clear and contains speech.
    The model works best with audio files that are 3-5 seconds in length.
    """)

# Add a sidebar with additional information
with st.sidebar:
    st.header("Speech Emotion Detection")
    st.write("This app uses a pre-trained TensorFlow model to detect emotions in speech.")
    st.write("Supported emotions:")
    for emotion in detector.emotions:
        st.write(f"- {emotion.capitalize()}")
    
    st.subheader("Technical Details")
    st.write("""
    The model uses MFCC (Mel-Frequency Cepstral Coefficients) features extracted from the audio.
    These features capture the tonal and rhythmic characteristics of speech that relate to emotional content.
    """)
    
    # Requirements note
    st.subheader("Requirements")
    st.code("""
    streamlit
    tensorflow
    numpy
    librosa
    """)
