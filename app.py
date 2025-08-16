import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Load Model & Data
# -----------------------------
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered",
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
        }
        .stButton button {
            width: 100%;
            border-radius: 12px;
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 0.6em;
            margin-top: 15px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .pred-box {
            padding: 20px;
            border-radius: 12px;
            background-color: #e8f5e9;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: #2e7d32;
            margin-top: 20px;
        }
        .stSelectbox, .stNumberInput, .stRadio, .stSlider {
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)


# -----------------------------
# Title
# -----------------------------
st.title("💻 Laptop Price Predictor")
st.write("Know the market value of your laptop before you buy or sell.")


# -----------------------------
# Input Form
# -----------------------------
with st.form("prediction_form"):

    st.subheader("📌 General Info")
    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox('Brand', df['Company'].unique())
        type_ = st.selectbox('Type', df['TypeName'].unique())
        ram = st.selectbox('RAM (GB)', [2,4,6,8,12,16,24,32,64])
    with col2:
        weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, step=0.1)
        cpu = st.selectbox('CPU', df['Cpu_brand'].unique())
        os = st.selectbox('Operating System', df['Os'].unique())

    st.subheader("🖥️ Display")
    col3, col4 = st.columns(2)
    with col3:
        touchscreen = st.radio('Touchscreen', ['No','Yes'], horizontal=True)
        ips = st.radio('IPS Display', ['No','Yes'], horizontal=True)
    with col4:
        screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.0)
        resolution = st.selectbox(
            'Screen Resolution',
            ['1920x1080','1366x768','1600x900','3840x2160',
             '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440']
        )

    st.subheader("💾 Storage & Graphics")
    col5, col6 = st.columns(2)
    with col5:
        hdd = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048])
        ssd = st.selectbox('SSD (GB)', [0,8,128,256,512,1024])
    with col6:
        gpu = st.selectbox('GPU', df['Gpu_brand'].unique())

    # Submit button inside form
    submit = st.form_submit_button("🔮 Predict Price")

# -----------------------------
# Prediction
# -----------------------------
if submit:  # ✅ use submit from the form
    # query
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    query = np.array([company, type_, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)

    # Predict log price → convert back
    predicted_price = int(np.exp(pipe.predict(query)[0]))

    # Round to nearest 1000
    lower = (predicted_price // 1000) * 1000
    upper = lower + 1000

    st.markdown(f"<div class='pred-box'>💰 Estimated Price Range: ₹{lower:,} - ₹{upper:,}</div>", unsafe_allow_html=True)
