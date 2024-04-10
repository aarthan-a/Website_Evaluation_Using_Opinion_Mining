import streamlit as st
import pandas as pd
import re
import nltk
import tensorflow as tf
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
import json
import plotly.express as px

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model
model = load_model('Emotion_Detection.h5')

# Load the tokenizer
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    data = f.read()
    tokenizer = tokenizer_from_json(data)

# Expanding Contractions
def cont_exp(x):
    contractions = {"you're": "you are", "i'm": "i am", "he's": "he is"}
    for key in contractions:
        x = x.replace(key, contractions[key])
    return x

# Define function to preprocess input text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = cont_exp(text)
    # Remove emails, HTML tags, special characters, and accented characters
    text = re.sub(r'\S*@\S*\s?|<.*?>|[^a-zA-Z\s]|[\xc3\xa1\xc3\xa9\xc3\xad\xc3\xb3\xc3\xba\xc3\xb1\xc3\x81\xc3\x89\xc3\x8d\xc3\x93\xc3\x9a\xc3\x91]', '', text)

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return ' '.join(tokens)

# Define function to predict emotions
def predict_emotion(text):
    emotions_count = {'anger': 0, 'fear': 0, 'joy': 0, 'love': 0, 'sadness': 0, 'surprise': 0}
    sentences = text.split('\n')  # Split input into sentences based on new lines
    
    for sentence in sentences:
        if sentence.strip() == "":
            continue
        preprocessed_text = preprocess_text(sentence)
        sequence = tokenizer.texts_to_sequences([preprocessed_text])
        padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')
        
        prediction = model.predict(padded_sequence)
        predicted_index = np.argmax(prediction)
        emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
        
        predicted_emotion = emotions[predicted_index]
        emotions_count[predicted_emotion] += 1

    return emotions_count

def main():
    st.title("Emotion Detection App")

    user_input = st.text_area("Enter your text (sentences separated by new lines):", "")

    if st.button("Predict Emotion"):
        if user_input.strip() != "":
            emotions_count = predict_emotion(user_input)
            df = pd.DataFrame(list(emotions_count.items()), columns=['Emotion', 'Count'])

            # Create a pie chart with Plotly Express
            fig = px.pie(df, names='Emotion', values='Count', title="Emotion Distribution",
                         color_discrete_sequence=px.colors.qualitative.Set3)

            # Optional: Customizing the pie chart
            fig.update_traces(textinfo='percent+label', pull=[0.1 if max(df['Count']) == count else 0 for count in df['Count']])
            fig.update_layout(legend_title_text='Emotion')

            st.plotly_chart(fig)
        else:
            st.write("Please enter some text to predict emotion.")

if __name__ == "__main__":
    main()