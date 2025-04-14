import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

class FaceEmotionDetector:
    def __init__(self, model_path="models/face_model.h5"):
        self.model = tf.keras.models.load_model(model_path)
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def preprocess_image(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, None
            
        # Get the first face
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        
        # Resize to model input size
        face = cv2.resize(face, (48, 48))
        
        # Normalize
        face = face / 255.0
        
        # Reshape for model input
        return face.reshape(1, 48, 48, 1), (x, y, w, h)
    
    def predict(self, image):
        processed_image, face_coords = self.preprocess_image(image)
        if processed_image is None:
            return "No face detected", None
            
        predictions = self.model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        return self.emotions[predicted_class], face_coords

# Set page configuration
st.set_page_config(
    page_title="Face Emotion Detection",
    page_icon="😊",
    layout="wide"
)

# App title and description
st.title("Face Emotion Detection")
st.write("Upload an image or take a photo to detect emotions.")

# Initialize the model
@st.cache_resource
def load_model():
    try:
        return FaceEmotionDetector()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

detector = load_model()

# Add file uploader and camera input options
option = st.radio("Choose input method:", ["Upload Image", "Use Camera"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Convert the uploaded file to an image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process the image
        if st.button("Detect Emotion"):
            with st.spinner("Detecting emotion..."):
                # Convert from RGB to BGR (OpenCV format)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                emotion, face_coords = detector.predict(img_array)
                
                if face_coords:
                    # Draw rectangle around face
                    x, y, w, h = face_coords
                    img_with_face = img_array.copy()
                    cv2.rectangle(img_with_face, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Add emotion text
                    cv2.putText(img_with_face, emotion, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    
                    # Convert back to RGB for display
                    img_with_face = cv2.cvtColor(img_with_face, cv2.COLOR_BGR2RGB)
                    st.image(img_with_face, caption=f"Detected Emotion: {emotion.capitalize()}", 
                             use_column_width=True)
                else:
                    st.warning("No face detected in the image.")

elif option == "Use Camera":
    st.write("Take a photo with your camera")
    camera_image = st.camera_input("Take a picture")
    
    if camera_image is not None:
        # Convert the camera input to an image
        image = Image.open(camera_image)
        img_array = np.array(image)
        
        # Process the image
        with st.spinner("Detecting emotion..."):
            # Convert from RGB to BGR (OpenCV format)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            emotion, face_coords = detector.predict(img_array)
            
            if face_coords:
                # Draw rectangle around face
                x, y, w, h = face_coords
                img_with_face = img_array.copy()
                cv2.rectangle(img_with_face, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add emotion text
                cv2.putText(img_with_face, emotion, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                
                # Convert back to RGB for display
                img_with_face = cv2.cvtColor(img_with_face, cv2.COLOR_BGR2RGB)
                st.image(img_with_face, caption=f"Detected Emotion: {emotion.capitalize()}", 
                         use_column_width=True)
            else:
                st.warning("No face detected in the image.")

# Add information about the app
with st.expander("About this app"):
    st.write("""
    This app detects emotions in faces using a machine learning model. 
    It can identify seven emotions: anger, disgust, fear, happiness, neutral, sadness, and surprise.
    
    To use the app:
    1. Choose whether to upload an image or take a photo with your camera
    2. If uploading, click 'Detect Emotion' after selecting your image
    3. The app will display the detected emotion and highlight the face in the image
    
    Note: For best results, ensure the face is clearly visible and well-lit.
    """)

# Add a sidebar with additional information
with st.sidebar:
    st.header("Emotion Detection")
    st.write("This app uses a pre-trained TensorFlow model to detect emotions in faces.")
    st.write("Supported emotions:")
    for emotion in detector.emotions:
        st.write(f"- {emotion.capitalize()}")
