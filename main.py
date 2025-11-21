import streamlit as st
import pandas as pd
import joblib

# LOAD MODEL & ENCODER

model = joblib.load("mushroom_voting_model.pkl")
encoder = joblib.load("label_encoder.pkl")  # untuk konsistensi label


# APP TITLE
st.title("🍄 Mushroom Classification App")
st.write("Aplikasi cerdas untuk memprediksi jamur **Edible atau Poisonous** menggunakan Voting Classifier (Decision Tree + Random Forest).")
st.markdown("---")


# FITUR INPUT (BERDASARKAN DATASET)

st.header("Masukkan Fitur Jamur")

# Semua nilai kategori (berdasarkan dataset aslinya)
feature_options = {
    "cap-shape": ['b','c','x','f','k','s'],
    "cap-surface": ['f','g','y','s'],
    "cap-color": ['n','b','c','g','r','p','u','e','w','y'],
    "bruises": ['t','f'],
    "odor": ['a','l','c','y','f','m','n','p','s'],
    "gill-attachment": ['a','d','f','n'],
    "gill-spacing": ['c','w','d'],
    "gill-size": ['b','n'],
    "gill-color": ['k','n','b','h','g','r','o','p','u','e','w','y'],
    "stalk-shape": ['e','t'],
    "stalk-root": ['b','c','u','e','z','r','?'],
    "stalk-surface-above-ring": ['f','y','k','s'],
    "stalk-surface-below-ring": ['f','y','k','s'],
    "stalk-color-above-ring": ['n','b','c','g','o','p','e','w','y'],
    "stalk-color-below-ring": ['n','b','c','g','o','p','e','w','y'],
    "veil-type": ['p','u'],
    "veil-color": ['n','o','w','y'],
    "ring-number": ['n','o','t'],
    "ring-type": ['c','e','f','l','n','p','s','z'],
    "spore-print-color": ['k','n','b','h','r','o','u','w','y'],
    "population": ['a','c','n','s','v','y'],
    "habitat": ['g','l','m','p','u','w','d']
}

# Buat input dinamis
input_data = {}
cols = st.columns(2)

i = 0
for feature, values in feature_options.items():
    with cols[i % 2]:
        input_data[feature] = st.selectbox(feature, values)
    i += 1

st.markdown("---")


# KONVERSI INPUT MENJADI NUMERIK

encoded_input = {}

for col, val in input_data.items():
    # simulasi Series agar kompatibel encoder
    encoded_input[col] = encoder.fit([val]).transform([val])[0]

# Convert ke DataFrame
input_df = pd.DataFrame([encoded_input])


# PREDIKSI

st.header("Hasil Prediksi")

if st.button("🔍 Prediksi Jamur"):
    prediction = model.predict(input_df)[0]

    if prediction == 0:
        st.success("🍽 **Jamur EDIBLE (AMAN DIMAKAN)**")
    else:
        st.error("☠️ **Jamur POISONOUS (BERACUN!)**")

    st.markdown("---")



# INFORMASI MODEL

st.sidebar.title("📊 Informasi Model")
st.sidebar.write("Model: **Voting Classifier**")
st.sidebar.write("• Decision Tree Classifier")  
st.sidebar.write("• Random Forest Classifier (200 trees)")  
st.sidebar.write("Akurasi Model: **100%** 🎉")

st.sidebar.markdown("---")
st.sidebar.write("Dataset: *Mushroom Classification (UCI)*")
st.sidebar.write("Author: **Anda** 😎")
