# model/train.py
import joblib
from sklearn.metrics import precision_score, recall_score
import numpy as np
from data_generator import generate_synthetic_data
from features import encode_features
from anomaly_model import train_anomaly_model
from fraud_model import train_fraud_model

def generate_and_train():
    print("Step 1: Generating Synthetic Banking Data...")
    df = generate_synthetic_data()

    print("Step 2: Preprocessing & Feature Engineering...")
    df, le = encode_features(df)

    X_base = df[['Amount', 'Time_of_Day', 'Distance_from_Home_km', 'Merchant_Type_Enc']]

    print("Step 3: Training Anomaly Detection Layer...")
    iso_forest, anomaly_scores = train_anomaly_model(X_base)
    df['Anomaly_Score'] = anomaly_scores

    X_hybrid = df[['Amount', 'Time_of_Day', 'Distance_from_Home_km', 'Merchant_Type_Enc', 'Anomaly_Score']]
    y = df['Fraud_Label']

    print("Step 4: Training Supervised Fraud Model...")
    xgb_model = train_fraud_model(X_hybrid, y)

    # STEP 5: Threshold tuning to reduce false positives
    probs = xgb_model.predict_proba(X_hybrid)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 9)
    best_threshold = 0.5
    best_precision = 0

    for t in thresholds:
        preds = (probs >= t).astype(int)
        precision = precision_score(y, preds)
        recall = recall_score(y, preds)

        # Prefer higher precision while keeping reasonable recall
        if precision > best_precision and recall >= 0.90:
            best_precision = precision
            best_threshold = t

    print(f"Selected decision threshold: {best_threshold:.2f}")
    print(f"Precision at threshold: {best_precision:.2f}")


    print("Step 6: Saving artifacts...")
    artifacts = {
    'iso_forest': iso_forest,
    'xgb_model': xgb_model,
    'label_encoder': le,
    'feature_cols': list(X_hybrid.columns),
    'decision_threshold': best_threshold
    }


    joblib.dump(artifacts, 'artifacts/fraud_model_bundle.joblib')
    print("Success! Artifacts saved.")

if __name__ == "__main__":
    generate_and_train()
