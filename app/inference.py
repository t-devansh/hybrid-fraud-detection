# app/inference.py
import pandas as pd
import joblib
import streamlit as st

@st.cache_resource
def load_engine():
    return joblib.load('artifacts/fraud_model_bundle.joblib')

def score_transaction(amount, time_of_day, distance, merchant_type):
    bundle = load_engine()
    threshold = bundle['decision_threshold']

    iso_forest = bundle['iso_forest']
    xgb_model = bundle['xgb_model']
    le = bundle['label_encoder']

    raw_input = pd.DataFrame(
        [[amount, time_of_day, distance, le.transform([merchant_type])[0]]],
        columns=[
            'Amount',
            'Time_of_Day',
            'Distance_from_Home_km',
            'Merchant_Type_Enc'
        ]
    )

    raw_input['Anomaly_Score'] = iso_forest.decision_function(raw_input)
    prob = xgb_model.predict_proba(raw_input)[0][1]

    return prob, raw_input, xgb_model, threshold

