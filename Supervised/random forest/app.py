import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model dan label encoder
with open('model_rf_ikan.pkl', 'rb') as file:
    model = pickle.load(file)
    
with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

# Fungsi untuk prediksi
def predict_fish(length, weight):
    ratio = weight/length
    data = pd.DataFrame({
        'length': [length],
        'weight': [weight],
        'w_l_ratio': [ratio]
    })
    prediction = model.predict(data)
    species = le.inverse_transform([prediction])[0]
    return species

# UI Aplikasi
st.title("🐟 Prediksi Spesies Ikan")
st.write("Masukkan pengukuran ikan untuk mengetahui spesiesnya")

# Input dalam 2 kolom
col1, col2 = st.columns(2)

with col1:
    length = st.number_input("Panjang Ikan (cm)", 
                           min_value=6.0, 
                           max_value=34.0, 
                           value=15.0)

with col2:
    weight = st.number_input("Berat Ikan (kg)", 
                           min_value=2.0, 
                           max_value=6.3, 
                           value=3.0)

# Tombol prediksi
if st.button("Prediksi"):
    species = predict_fish(length, weight)
    st.success(f"Spesies Ikan: **{species}**")
    st.write(f"Rasio Berat/Panjang: {(weight/length):.3f}")

# Informasi singkat
with st.expander("ℹ️ Informasi Dataset"):
    st.write("""
    Dataset mencakup 9 spesies ikan:
    - Anabas testudineus
    - Coilia dussumieri
    - Otolithoides biauritus
    - Otolithoides pama
    - Pethia conchonius
    - Polynemus paradiseus
    - Puntius lateristriga
    - Setipinna taty
    - Sillaginopsis panijus
    """)
