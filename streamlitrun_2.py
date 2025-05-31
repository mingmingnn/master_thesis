import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load BERT models
sentiment_model_path = "./bert_sentiment_model"
sentiment_tokenizer = BertTokenizer.from_pretrained(sentiment_model_path)
sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_path)
sentiment_model.eval()

reason_model_path = "./bert_feedback_final_model"
reason_tokenizer = BertTokenizer.from_pretrained(reason_model_path)
reason_model = BertForSequenceClassification.from_pretrained(reason_model_path)
reason_model.eval()

# Load MLP models and scaler
mlp_model_c = load_model("mlp_comfort_model.keras")
mlp_model_s = load_model("mlp_satisfaction_model.keras")
scaler = joblib.load("./scaler.save")
onehot_template = pd.read_csv("onehot_template.csv")

# NLP model inference
def predict_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = sentiment_model(**inputs).logits
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).item()
    return pred

def predict_reason(text):
    inputs = reason_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = reason_model(**inputs).logits
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).item()
    return pred

# Streamlit UI
st.title("Hearing Aid Comfort & Satisfaction Predictor")
st.markdown("Please enter the user's profile and discomfort description. The model will predict both comfort score (regression) and satisfaction level (classification).")

zone_map = {
    "Lower-middle region": -1,
    "Upper-left region": 0,
    "Upper-right region": 1,
    "Ear Canal entrance": 2,
    "Lower-left scattered": 3,
    "Bottom-right edge": 4
}

# User Input
col1, col2 = st.columns(2)

with col1:
    age = st.selectbox("Age range", ["10-14", "15-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "Over 90"])
    gender = st.selectbox("Gender", ["Male", "Female", "Other (please specify)", "Prefer not to answer"])
    style = st.selectbox("Hearing Aid Style", [
        "miniRITE R", "miniRITE T", "miniRITE", "miniBTE T", "miniBTE R",
        "BTE PP", "BTE UP", "BTE SP", "ITE, ITC", "ITE, HS/FS",
        "ITE, IIC/CIC", "ITE,"
    ])

with col2:
    config = st.selectbox("Earpiece Configuration", ["Domes", "Custom", "Custom/Dome", "Other"])
    used_years = st.selectbox("How long has the user been using hearing aids?", ["Less than 1 year", "1-2 years", "2-3 years", "3-5 years", "More than 5 years"])
    platform = st.selectbox("Platform", ["Opn/Opn S", "More", "Real"])


zone_label = st.selectbox("Ear click region", list(zone_map.keys()))
zone = zone_map[zone_label]

# Text Input
discomfort = st.text_area("Describe the discomfort the user is experiencing:")
suggestion = st.text_area("What can be done to improve comfort?")

if st.button("Run Prediction"):
    
    # NLP predictions
    sentiment_label = predict_sentiment(discomfort)
    reason_label = predict_reason(discomfort)

    # Map predicted discomfort category
    reason_map = {
        0: "Dome Size Issue",
        1: "Improper Fit (Ear Canal Pressure)",
        2: "Dome-Wire Irritation",
        3: "Stability Issue and Wearing Scenario Discomfort",
        4: "Interfere with Glasses",
        5: "No Discomfort",
        6: "Severe Itching and Foreign Body Sensation",
        7: "General Itching (Unspecified Itching Cause)"
    }
    reason_text = reason_map.get(reason_label, "Unknown")

    # Structured input
    data = {
        "Style": style,
        "Platform": platform,
        "Earpiece Configuration": config,
        "What is your gender?": gender,
        "What is your current age?": age,
        "How long have you been using hearing aids?": used_years,
        "sentiment_label": sentiment_label,
        "normalized_label": reason_label,
        "zone_cluster": zone,
        "Slip Out": 0,
        "Annoying": 0,
        "Change Position": 0,
        "Too tight": 0,
        "Itchiness": 0,
        "Soreness": 0,
        "Take off hearing aids": 0,
        "Painful": 0
    }

    # Preprocess for model
    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=onehot_template.columns, fill_value=0)
    X_scaled = scaler.transform(input_df)

    # Predictions
    comfort_pred = mlp_model_c.predict(X_scaled).flatten()[0]
    satisfaction_probs = mlp_model_s.predict(X_scaled)
    satisfaction_class = np.argmax(satisfaction_probs, axis=1)[0]
    satisfaction_label = ["Dissatisfied", "Neutral", "Satisfied"][satisfaction_class]

    # Output
    st.success(f"Predicted Comfort Score: **{comfort_pred:.2f}**")
    st.success(f"Predicted Satisfaction Level: **{satisfaction_label}**")
    st.info(f"Model-identified discomfort type: **{reason_text}**")
