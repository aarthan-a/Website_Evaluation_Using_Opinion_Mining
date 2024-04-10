import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import re
import numpy as np
import pandas as pd
import plotly.express as px

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model.eval()

def preprocess_text_before_tokenization(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    return text

def predict_sentiment(text):
    preprocessed_text = preprocess_text_before_tokenization(text)
    inputs = tokenizer.encode_plus(
        preprocessed_text, return_tensors='pt', max_length=512, truncation=True, padding='max_length', add_special_tokens=True
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    sentiment_scores = torch.softmax(logits, dim=1).detach().cpu().numpy()
    predicted_class_id = np.argmax(sentiment_scores, axis=1)[0]

    sentiments = ['Negative', 'Neutral', 'Positive']
    sentiment = sentiments[predicted_class_id]
    return sentiment

def main():
    st.title("Sentiment Analysis with RoBERTa")

    user_input = st.text_area("Enter texts (each text on a new line):")

    if st.button('Analyze Sentiment'):
        texts = user_input.split('\n')
        sentiment_counts = {'Negative': 0, 'Neutral': 0, 'Positive': 0}
        
        for text in texts:
            if text.strip():  # Ignore empty lines
                sentiment = predict_sentiment(text)
                sentiment_counts[sentiment] += 1

        # Convert sentiment counts to a DataFrame for visualization
        df_sentiments = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment', 'Count'])

        # Display bar chart using Plotly
        fig = px.bar(df_sentiments, x='Sentiment', y='Count', color='Sentiment', title='Sentiment Distribution')
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
