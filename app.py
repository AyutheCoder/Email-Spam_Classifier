import nltk
import os
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))
nltk.download('punkt', download_dir='nltk_data')
nltk.download('stopwords', download_dir='nltk_data')
# Download NLTK data if missing (must be at the very top)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Modern glassmorphism style: vibrant gradient, glassy box, green accent for input

def set_bg_and_style():
    st.markdown(
        """
        <style>
        body, .stApp {
            background: #0a2342 !important;
        }
        .stTextArea textarea {
            background: #10284e !important;
            border: 2px solid #27ae60 !important;
            color: #fff !important;
            border-radius: 10px !important;
            font-size: 1.1em;
            caret-color: #fff !important;
        }
        .stTextArea label {
            color: #fff !important;
            font-weight: bold;
        }
        .stButton button {
            background: #27ae60 !important;
            color: #fff !important;
            border-radius: 8px !important;
            border: none !important;
            font-weight: bold;
            font-size: 1.1em;
        }
        .stButton button:hover {
            background: #219150 !important;
        }
        .glass-box {
            background: #27ae60;
            border-radius: 18px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.25);
            padding: 32px 24px;
            margin: 32px 0 0 0;
            text-align: center;
            border: 2px solid #1e824c;
        }
        .glass-box h2 {
            color: #fff !important;
            font-size: 2.2em;
            margin: 0;
        }
        .custom-title {
            color: #fff !important;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 0.5em;
            text-align: center;
            letter-spacing: 1px;
        }
        .stTitle, .stTitle h1, .stTitle h2, .stTitle h3, .stTitle h4, .stTitle h5, .stTitle h6 {
            color: #fff !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_and_style()

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = re.findall(r'\b\w+\b', text)  # safe, regex-based tokenizer
    y = []
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Custom white title
st.markdown('<div class="custom-title">Email/SMS Spam Classifier</div>', unsafe_allow_html=True)

input_sms = st.text_area("Enter the message", key="input_sms")

if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display with green highlight box and white text
    if result == 1:
        st.markdown('<div class="glass-box"><h2>Spam</h2></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-box"><h2>Not Spam</h2></div>', unsafe_allow_html=True)
