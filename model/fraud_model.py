# model/fraud_model.py
from xgboost import XGBClassifier

def train_fraud_model(X, y):
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X, y)
    return model
