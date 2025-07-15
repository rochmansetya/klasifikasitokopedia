import streamlit as st
import pickle
import nltk
import re
import string

# Load model (pipeline: vectorizer + classifier)
with open('model_naive_bayes.pkl', 'rb') as file:
    model = pickle.load(file)

# Preprocessing fungsi ringan
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# UI
st.title("Klasifikasi Sentimen Review Tokopedia")

user_input = st.text_area("Masukkan review produk:")

if st.button("Prediksi"):
    if user_input:
        clean_text = preprocess_text(user_input)
        prediction = model.predict([clean_text])[0]
        st.success(f"Prediksi Sentimen: {prediction}")
    else:
        st.warning("Masukkan teks terlebih dahulu.")
