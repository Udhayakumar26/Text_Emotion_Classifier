# import pandas as pd
# import numpy as np
# import keras
# import tensorflow
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Embedding, Flatten, Dense
# data = pd.read_csv("train.txt", sep=';')
# data.columns = ["Text", "Emotions"]
# data.head()
# texts = data["Text"].tolist()
# labels = data["Emotions"].tolist()
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(texts)
# label_encoder = LabelEncoder()
# labels = label_encoder.fit_transform(labels)
# one_hot_labels = keras.utils.to_categorical(labels)
# xtrain, xtest, ytrain, ytest = train_test_split(padded_sequences,  one_hot_labels,  test_size=0.2)
# model = Sequential()
# model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length))
# model.add(Flatten())
# model.add(Dense(units=128, activation="relu"))
# model.add(Dense(units=len(one_hot_labels[0]), activation="softmax"))
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.fit(xtrain, ytrain, epochs=15, batch_size=32, validation_data=(xtest, ytest))
# model_architecture = model.to_json()
# with open("model_architecture.json", "w") as json_file:json_file.write(model_architecture)
# model.save_weights("model.weights.h5")
# # input_text = " i feel well too "

# def change(input_text):
#     input_sequence = tokenizer.texts_to_sequences([input_text])
#     padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
#     prediction = model.predict(padded_input_sequence)
#     predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
#     return (predicted_label)
# print(change('i feel well too'))


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

# Load the data
data = pd.read_csv("train.txt", sep=';')
data.columns = ["Text", "Emotions"]

# Display first few rows of the data
print(data.head())

# Prepare the texts and labels
texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

# Initialize the tokenizer and fit on texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Tokenize the texts
sequences = tokenizer.texts_to_sequences(texts)

# Define max_length for padding (you can choose a value)
max_length = 100  # Example value, adjust based on your dataset

# Pad the tokenized sequences
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Convert the labels to one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels)

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(padded_sequences, one_hot_labels, test_size=0.2)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=len(one_hot_labels[0]), activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(xtrain, ytrain, epochs=15, batch_size=32, validation_data=(xtest, ytest))

# Save the model architecture and weights
model_architecture = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_architecture)
model.save_weights("model.weights.h5")

# Define a function to predict the emotion of the input text
def change(input_text):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
    prediction = model.predict(padded_input_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
    return predicted_label

# Test the prediction function
print(change('i feel well too'))
