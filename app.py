import streamlit as st
import joblib  
import nltk
import re
import string

# Load model (gunakan joblib, karena kamu pakai joblib.dump)
model = joblib.load("model_naive_bayes.pkl")

# Fungsi preprocessing ringan
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# UI Streamlit
st.title("Klasifikasi Sentimen Review")

user_input = st.text_area("Masukkan review produk:")

if st.button("Prediksi"):
    if user_input:
        clean_text = preprocess_text(user_input)
        prediction = model.predict([clean_text])[0]
        st.success(f"Prediksi Sentimen: {'Positif' if prediction == 1 else 'Negatif'}")
    else:
        st.warning("Masukkan teks terlebih dahulu.")
