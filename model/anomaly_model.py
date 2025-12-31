# model/anomaly_model.py
from sklearn.ensemble import IsolationForest

def train_anomaly_model(X, contamination=0.01):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_scores = iso_forest.fit(X).decision_function(X)
    return iso_forest, anomaly_scores
