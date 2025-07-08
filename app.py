from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import os
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model components
tokenizer = None
label_encoder = None
model = None
max_length = None

def load_model_components():
    """Load and prepare tokenizer, encoder, and model"""
    global tokenizer, label_encoder, model, max_length
    
    try:
        # Check if required files exist
        required_files = ["train.txt", "model_architecture.json", "model.weights.h5"]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False
        
        # Load and prepare data
        logger.info("Loading training data...")
        data = pd.read_csv("train.txt", sep=';')
        data.columns = ["Text", "Emotions"]
        
        texts = data["Text"].tolist()
        labels = data["Emotions"].tolist()
        
        # Prepare tokenizer
        logger.info("Preparing tokenizer...")
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        max_length = max(len(seq) for seq in sequences)
        
        # Prepare label encoder
        logger.info("Preparing label encoder...")
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        
        # Load model
        logger.info("Loading model...")
        with open("model_architecture.json", "r") as json_file:
            loaded_model_json = json_file.read()
        
        model = model_from_json(loaded_model_json)
        model.load_weights("model.weights.h5")
        
        logger.info("Model components loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        return False

def predict_emotion(text):
    """Predict emotion for given text"""
    try:
        if not all([tokenizer, label_encoder, model, max_length]):
            return None, 0.0
        
        # Preprocess input
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=max_length)
        
        # Predict
        prediction = model.predict(padded)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        
        # Get emotion label
        predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
        
        return predicted_emotion, confidence
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None, 0.0

def get_emotion_icon(emotion):
    """Get emoji icon for emotion"""
    emotion_icons = {
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'fear': '😨',
        'surprise': '😲',
        'disgust': '🤢',
        'neutral': '😐',
        'love': '❤️',
        'excitement': '🎉',
        'anxiety': '😰',
        'joy': '😄',
        'worry': '😟'
    }
    return emotion_icons.get(emotion.lower(), '😐')

# === Routes ===
@app.route("/", methods=["GET"])
def home():
    """Home page with input form"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict emotion from form submission"""
    try:
        input_text = request.form.get("input_text", "").strip()
        
        # Validate input
        if not input_text:
            return render_template("index.html", error="Please enter some text to analyze.")
        
        if len(input_text) > 500:
            return render_template("index.html", error="Text is too long. Please limit to 500 characters.")
        
        # Check if model is loaded
        if not all([tokenizer, label_encoder, model, max_length]):
            return render_template("index.html", error="Model not loaded. Please try again later.")
        
        # Predict emotion
        predicted_emotion, confidence = predict_emotion(input_text)
        
        if predicted_emotion is None:
            return render_template("index.html", error="Error occurred during prediction. Please try again.")
        
        # Get additional data for result page
        emotion_icon = get_emotion_icon(predicted_emotion)
        confidence_percentage = round(confidence * 100, 1)
        
        return render_template("result.html", 
                             input_text=input_text, 
                             predicted_emotion=predicted_emotion,
                             emotion_icon=emotion_icon,
                             confidence=confidence_percentage)
        
    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        return render_template("index.html", error="An unexpected error occurred. Please try again.")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for emotion prediction (for AJAX calls)"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        input_text = data['text'].strip()
        
        if not input_text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        if len(input_text) > 500:
            return jsonify({'error': 'Text too long (max 500 characters)'}), 400
        
        # Check if model is loaded
        if not all([tokenizer, label_encoder, model, max_length]):
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Predict emotion
        predicted_emotion, confidence = predict_emotion(input_text)
        
        if predicted_emotion is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        return jsonify({
            'emotion': predicted_emotion,
            'confidence': round(confidence * 100, 1),
            'icon': get_emotion_icon(predicted_emotion)
        })
        
    except Exception as e:
        logger.error(f"Error in API predict: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    model_loaded = all([tokenizer, label_encoder, model, max_length])
    
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'max_length': max_length if max_length else 0
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template("index.html", error="Page not found."), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template("index.html", error="Internal server error."), 500

# === Initialize and Run App ===
if __name__ == "__main__":
    logger.info("Starting Emotion Classifier App...")
    
    # Load model components
    if load_model_components():
        logger.info("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to load model components. Please check your files and try again.")
        print("\n" + "="*50)
        print("❌ STARTUP ERROR")
        print("="*50)
        print("Required files:")
        print("  - train.txt")
        print("  - model_architecture.json") 
        print("  - model.weights.h5")
        print("\nPlease ensure all files are in the same directory as app.py")
        print("="*50)