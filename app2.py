import streamlit as st
import pickle
import nltk
import os
import string

# ✅ NLTK Data Setup (local folder)
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_path)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Load vectorizer and model
cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Original NLTK preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Streamlit UI
st.title("📩 SMS Spam Classifier (NLTK Version)")
st.write("This version uses NLTK for text preprocessing.")

input_sms = st.text_area("✉️ Enter your message:")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Preprocess the message
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector = cv.transform([transformed_sms]).toarray()

        # Predict
        result = model.predict(vector)[0]

        # Display
        if result == 1:
            st.error("🚨 This message is **SPAM**.")
        else:
            st.success("✅ This message is **NOT SPAM**.")
