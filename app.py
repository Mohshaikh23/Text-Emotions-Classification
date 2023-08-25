import streamlit as st
import pandas as pd
import numpy as np
import keras
import tensorflow
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the data
data = pd.read_csv("train.txt", sep=';')
data.columns = ["Text", "Emotions"]

texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
max_length = max([len(seq) for seq in sequences])

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Load the saved model architecture
json_file = open("model_architecture.json", "r")
loaded_model_json = json_file.read()
json_file.close()

# Load the saved model weights
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_weights.h5")

# Streamlit app
def main():
    st.title("Text Emotion Classification")

    # User input
    input_text = st.text_area("Enter a sentence:", "")

    if st.button("Classify Emotion"):
        if input_text:
            input_sequence = tokenizer.texts_to_sequences([input_text])
            padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
            prediction = loaded_model.predict(padded_input_sequence)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
            st.write(f"Predicted Emotion: {predicted_label}")

if __name__ == "__main__":
    main()
