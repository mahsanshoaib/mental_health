import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class TextEmotionDetector:
    def __init__(self, model_path="text_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Define emotion labels (adjust based on your model's output)
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
    
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()
            confidence = predictions[0][predicted_class].item()
            
        return self.emotions[predicted_class], confidence

# Set page configuration
st.set_page_config(
    page_title="Text Emotion Detection",
    page_icon="📝",
    layout="wide"
)

# App title and description
st.title("Text Emotion Detection")
st.write("Enter text to detect the emotion.")

# Initialize the model
@st.cache_resource
def load_model():
    try:
        return TextEmotionDetector()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

detector = load_model()

# Text input
st.subheader("Enter Text")
user_text = st.text_area("Type or paste text here:", height=150)

# Process text when button is clicked
if st.button("Detect Emotion") and user_text:
    with st.spinner("Analyzing text..."):
        try:
            emotion, confidence = detector.predict(user_text)
            
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
            st.error(f"Error analyzing text: {e}")

# Add information about the app
with st.expander("About this app"):
    st.write("""
    This app detects emotions in text using a machine learning model. 
    It can identify seven emotions: anger, disgust, fear, happiness, neutral, sadness, and surprise.
    
    To use the app:
    1. Enter or paste text in the text area
    2. Click 'Detect Emotion' to analyze the text
    3. The app will display the detected emotion and confidence level
    
    Note: For best results, enter text that clearly expresses an emotional state.
    """)

# Add a sidebar with additional information
with st.sidebar:
    st.header("Text Emotion Detection")
    st.write("This app uses a pre-trained transformer model to detect emotions in text.")
    st.write("Supported emotions:")
    for emotion in detector.emotions:
        st.write(f"- {emotion.capitalize()}")
    
    st.subheader("Technical Details")
    st.write("""
    The model uses a transformer-based architecture from the Hugging Face library.
    It analyzes the semantic and contextual information in the text to identify emotional content.
    """)
    
    # Requirements note
    st.subheader("Requirements")
    st.code("""
    streamlit
    transformers
    torch
    """)
