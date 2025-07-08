# Text_Emotion_Classifier
# Text Emotion Classification using Machine Learning and Keras

This project is a web-based application that classifies the emotional tone behind a piece of text (e.g., joy, anger, sadness, etc.) using a deep learning model built with Keras. It features a user-friendly front end and a Flask backend to serve the predictions.

#required libraries :
 

- numpy
- pandas
- keras
- tensorflow
- scikit-learn
  
## Project Overview

-  Input: User enters a sentence in natural language.
-  Output: The predicted emotion label (e.g., joy, sadness, fear).
-  Model: Trained on labeled text data using a neural network built with TensorFlow/Keras.
-  Frontend: HTML/CSS with two pages – `index.html` for input and `result.html` for output.
-  Backend: Flask handles requests, prediction logic, and renders templates.

## 📁 Project Structure

├── app.py # Flask application
├── train.txt # Training dataset (Text;Emotion)
├── model_architecture.json # Saved Keras model structure
├── model.weights.h5 # Trained model weights
├── templates/
│ ├── index.html # User input form
│ └── result.html # Prediction display
├── static/ # (Optional) CSS or images
