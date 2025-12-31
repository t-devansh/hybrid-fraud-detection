# app/app.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from inference import score_transaction
from explainability import render_shap_explanation

st.set_page_config(page_title="Hybrid Fraud Monitor", layout="wide")
st.title("🛡️ Hybrid Financial Fraud Detection System")

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("Transaction Input")
amt = st.sidebar.number_input("Amount ($)", value=50.0)
tod = st.sidebar.slider("Time (Hour)", 0, 23, 12)
dist = st.sidebar.number_input("Distance (km)", value=2.0)
merchant = st.sidebar.selectbox("Merchant", ['Grocery', 'Gas', 'Online', 'Retail'])

# -----------------------------
# INFERENCE
# -----------------------------
prob, raw_input, model, threshold = score_transaction(
    amt, tod, dist, merchant
)

# -----------------------------
# 2×2 GRID LAYOUT
# -----------------------------
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# =============================
# BLOCK 1: VERDICT & METRICS
# =============================
with row1_col1:
    st.subheader("Verdict")

    if prob >= threshold:
        st.error("🚩 HIGH RISK")
    else:
        st.success("✅ SAFE")

    st.metric("Fraud Probability", f"{prob:.2%}")
    st.metric("Decision Threshold", f"{threshold:.2f}")
    st.metric("Anomaly Score", round(raw_input['Anomaly_Score'].values[0], 4))

# =============================
# BLOCK 2: SHAP EXPLAINABILITY (MOVED UP)
# =============================
with row1_col2:
    st.subheader("Why was this flagged? (SHAP)")
    fig_shap = render_shap_explanation(model, raw_input)
    st.pyplot(fig_shap, bbox_inches="tight")

# =============================
# BLOCK 3: RISK GAUGE (MOVED DOWN)
# =============================
with row2_col1:
    st.subheader("Risk Gauge")

    fig, ax = plt.subplots(figsize=(6, 1.6))
    ax.barh([0], [prob], color="red" if prob >= threshold else "green")
    ax.axvline(threshold, color="black", linestyle="--", label="Threshold")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Fraud Probability")
    ax.legend()

    st.pyplot(fig, bbox_inches="tight")

# =============================
# BLOCK 4: TRANSACTION COMPARISON
# =============================
with row2_col2:
    st.subheader("Transaction vs Typical Behavior")

    features = ['Amount ($)', 'Distance (km)']
    txn_vals = [amt, dist]
    baseline_vals = [50, 5]  # representative normal behavior

    x = np.arange(len(features))
    width = 0.35

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.bar(x - width/2, txn_vals, width, label="This Transaction")
    ax2.bar(x + width/2, baseline_vals, width, label="Typical Behavior")
    ax2.set_xticks(x)
    ax2.set_xticklabels(features)
    ax2.legend()

    st.pyplot(fig2, bbox_inches="tight")

# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.caption(
    "Hybrid fraud detection dashboard using anomaly detection, "
    "threshold-tuned decisioning, and explainable AI."
)
