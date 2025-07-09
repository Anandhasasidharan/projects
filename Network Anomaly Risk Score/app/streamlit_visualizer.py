import streamlit as st
import pandas as pd
import joblib
import shap

st.set_page_config(page_title="Cyber Threat Risk Scorer", layout="wide")

model = joblib.load("models/risk_score_model.pkl")
feature_names = pd.read_csv("data/feature_columns.csv")['feature'].tolist()

st.title("🔐 Threat Severity Predictor")

uploaded_file = st.file_uploader("📁 Upload CSV with traffic features", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    input_encoded = pd.get_dummies(input_df)

    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[feature_names]
    preds = model.predict(input_encoded)

    st.subheader("📊 Predicted Severity")
    input_df['Severity'] = preds
    st.dataframe(input_df)

    explainer = shap.Explainer(model)
    shap_values = explainer(input_encoded)
    st.pyplot(shap.plots.beeswarm(shap_values[:100]))
else:
    st.info("Upload a preprocessed CSV file to begin.")
