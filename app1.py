import streamlit as st
import pickle
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

ps = PorterStemmer()

cv = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

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

st.title("SMS Spam Classifier 🚀")
input_sms = st.text_area("Enter your message:")

if st.button('Predict'):

    transformed_sms = transform_text(input_sms)

    vector = cv.transform([transformed_sms]).toarray()

    result = model.predict(vector)[0]

    if result == 1:
        st.warning("⚠️ Spam Message")
    else:
        st.success("✅ Not Spam")
