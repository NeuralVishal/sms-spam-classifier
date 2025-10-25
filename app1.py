import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load vectorizer and model
cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Text preprocessing function (no NLTK)
def transform_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove punctuation
    words = text.split()  # tokenize
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]  # remove stopwords
    return " ".join(words)

# Streamlit UI
st.title("📩 SMS Spam Classifier")
st.write("Enter a message below to check if it's Spam or Not Spam.")

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
