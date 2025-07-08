import pandas as pd
import numpy as np
import keras
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense



data = pd.read_csv("train.txt", sep=';')
data.columns = ["Text", "Emotions"]
data.head()


texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)


sequences = tokenizer.texts_to_sequences(texts)
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length)


# Encode the string labels to integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)


def predict_emotion(input_text):
    with open("model_architecture.json", "r") as json_file:
        model_json = json_file.read()
    model = keras.models.model_from_json(model_json)
    model.load_weights("model.weights.h5")
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
    prediction = model.predict(padded_input_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
    return predicted_label[0]

