import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import cv2
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tempfile
import os
from PIL import Image
import io

st.set_page_config(page_title="Emotion Detection App", layout="wide")

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .result-text {
        font-size: 1.8rem;
        font-weight: 600;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 2rem;
    }
    .emotion-anger {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    .emotion-disgust {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .emotion-fear {
        background-color: #E0E7FF;
        color: #3730A3;
    }
    .emotion-happiness {
        background-color: #FEF3C7;
        color: #92400E;
    }
    .emotion-neutral {
        background-color: #F3F4F6;
        color: #1F2937;
    }
    .emotion-sadness {
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    .emotion-surprise {
        background-color: #FEE2E2;
        color: #9D174D;
    }
</style>
""", unsafe_allow_html=True)

class SpeechEmotionDetector:
    def __init__(self, model_path="models/speech_model.h5"):
        # In a real app, we'd load the actual model
        # For this example, we'll simulate the model
        # self.model = tf.keras.models.load_model(model_path)
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
        try:
            features = self.extract_features(audio_path)
            # In a real app, we'd use:
            # predictions = self.model.predict(features)
            # predicted_class = np.argmax(predictions[0])
            
            # For demonstration, we'll return a random emotion
            predicted_class = np.random.randint(0, len(self.emotions))
            return self.emotions[predicted_class]
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            return "neutral"  # Default fallback

class FaceEmotionDetector:
    def __init__(self, model_path="models/face_model.h5"):
        # In a real app, we'd load the actual model
        # self.model = tf.keras.models.load_model(model_path)
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def preprocess_image(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
            
        # Get the first face
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        
        # Resize to model input size
        face = cv2.resize(face, (48, 48))
        
        # Normalize
        face = face / 255.0
        
        # Reshape for model input
        return face.reshape(1, 48, 48, 1)
    
    def predict(self, image):
        try:
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                st.warning("No face detected in the image.")
                return "neutral"  # Default emotion if no face detected
                
            # In a real app, we'd use:
            # predictions = self.model.predict(processed_image)
            # predicted_class = np.argmax(predictions[0])
            
            # For demonstration, we'll return a random emotion
            predicted_class = np.random.randint(0, len(self.emotions))
            return self.emotions[predicted_class]
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return "neutral"  # Default fallback

class TextEmotionDetector:
    def __init__(self, model_path="models/text_model"):
        # In a real app, we'd load the actual model
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # self.model.to(self.device)
        # self.model.eval()
        
        # Define emotion labels
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
    
    def predict(self, text):
        try:
            # In a real app, we'd use:
            # inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            # inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # with torch.no_grad():
            #     outputs = self.model(**inputs)
            #     predictions = torch.softmax(outputs.logits, dim=1)
            #     predicted_class = torch.argmax(predictions, dim=1).item()
            
            # For demonstration, we'll return a random emotion
            predicted_class = np.random.randint(0, len(self.emotions))
            return self.emotions[predicted_class]
        except Exception as e:
            st.error(f"Error processing text: {e}")
            return "neutral"  # Default fallback

def get_emotion_emoji(emotion):
    emoji_map = {
        "anger": "😠",
        "disgust": "🤢",
        "fear": "😨",
        "happiness": "😀",
        "neutral": "😐",
        "sadness": "😢",
        "surprise": "😲"
    }
    return emoji_map.get(emotion, "❓")

def main():
    # Initialize the detectors
    speech_detector = SpeechEmotionDetector()
    face_detector = FaceEmotionDetector()
    text_detector = TextEmotionDetector()
    
    # App Layout
    st.markdown("<h1 class='main-header'>Multimodal Emotion Detection</h1>", unsafe_allow_html=True)
    
    # Input type selection
    st.markdown("<h2 class='sub-header'>Select Input Type</h2>", unsafe_allow_html=True)
    input_type = st.radio("", ["Image (Face)", "Text", "Voice (Audio)"], horizontal=True)
    
    # Result placeholder
    result_placeholder = st.empty()
    
    if input_type == "Image (Face)":
        st.markdown("<h2 class='sub-header'>Upload an Image</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        col1, col2 = st.columns(2)
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            col1.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert PIL Image to OpenCV format
            img_array = np.array(image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Detect emotion
            emotion = face_detector.predict(img_cv)
            
            # Display result
            emoji = get_emotion_emoji(emotion)
            col2.markdown(f"<div class='result-text emotion-{emotion}'>Detected Emotion: {emotion.capitalize()} {emoji}</div>", unsafe_allow_html=True)
            
            # Display additional information
            with col2:
                st.markdown("### Emotion Analysis")
                st.write(f"The facial expression in this image appears to convey {emotion}.")
                
                # Emotion descriptions
                emotion_descriptions = {
                    "anger": "Signs of frustration or displeasure are detected in the facial features.",
                    "disgust": "The facial expression shows signs of aversion or repulsion.",
                    "fear": "Facial features indicate apprehension or concern.",
                    "happiness": "A positive expression is detected, showing signs of joy or contentment.",
                    "neutral": "No strong emotional expression is detected in the facial features.",
                    "sadness": "The facial expression conveys a sense of unhappiness or melancholy.",
                    "surprise": "Features show a reaction of astonishment or shock."
                }
                
                st.write(emotion_descriptions.get(emotion, ""))
    
    elif input_type == "Text":
        st.markdown("<h2 class='sub-header'>Enter Text</h2>", unsafe_allow_html=True)
        text_input = st.text_area("Type your text here", height=150)
        
        if text_input:
            # Detect emotion
            emotion = text_detector.predict(text_input)
            
            # Display result
            emoji = get_emotion_emoji(emotion)
            result_placeholder.markdown(f"<div class='result-text emotion-{emotion}'>Detected Emotion: {emotion.capitalize()} {emoji}</div>", unsafe_allow_html=True)
            
            # Display additional information
            st.markdown("### Text Analysis")
            st.write(f"The sentiment analysis of your text indicates {emotion}.")
            
            # Text analysis descriptions
            emotion_descriptions = {
                "anger": "The text contains language that suggests frustration or displeasure.",
                "disgust": "The language in the text conveys aversion or repulsion.",
                "fear": "The text expresses concerns, worries, or apprehension.",
                "happiness": "The text has a positive tone, expressing joy or contentment.",
                "neutral": "The text doesn't contain strong emotional indicators.",
                "sadness": "The language suggests unhappiness or melancholy.",
                "surprise": "The text expresses astonishment or unexpected reactions."
            }
            
            st.write(emotion_descriptions.get(emotion, ""))
    
    else:  # Voice (Audio)
        st.markdown("<h2 class='sub-header'>Upload an Audio File</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
        
        if uploaded_file is not None:
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_path = tmp_file.name
            
            # Display audio player
            st.audio(uploaded_file, format='audio/wav')
            
            # Detect emotion
            emotion = speech_detector.predict(audio_path)
            
            # Delete the temporary file
            os.unlink(audio_path)
            
            # Display result
            emoji = get_emotion_emoji(emotion)
            result_placeholder.markdown(f"<div class='result-text emotion-{emotion}'>Detected Emotion: {emotion.capitalize()} {emoji}</div>", unsafe_allow_html=True)
            
            # Display additional information
            st.markdown("### Audio Analysis")
            st.write(f"The voice tone analysis indicates {emotion}.")
            
            # Audio analysis descriptions
            emotion_descriptions = {
                "anger": "The audio contains vocal patterns associated with frustration or displeasure.",
                "disgust": "The voice tone suggests aversion or repulsion.",
                "fear": "Vocal patterns indicate apprehension or concern.",
                "happiness": "The voice has tonal qualities associated with joy or contentment.",
                "neutral": "No strong emotional indicators are detected in the voice pattern.",
                "sadness": "The voice tone suggests unhappiness or melancholy.",
                "surprise": "The vocal pattern indicates astonishment or shock."
            }
            
            st.write(emotion_descriptions.get(emotion, ""))
    
    # Footer
    st.markdown("---")
    st.markdown("### About This App")
    st.write("""
    This application uses machine learning models to detect emotions across different modalities:
    - **Image**: Analyzes facial expressions to determine emotional state
    - **Text**: Processes written content to identify sentiment and emotion
    - **Voice**: Examines audio characteristics to recognize emotional tones
    
    Please note that emotion detection is not always 100% accurate and should be interpreted as an estimation.
    """)

if __name__ == "__main__":
    main()
