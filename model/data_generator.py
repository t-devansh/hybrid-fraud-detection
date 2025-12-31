# model/data_generator.py
import pandas as pd
import numpy as np

def generate_synthetic_data(n=10000, seed=42):
    np.random.seed(seed)

    data = {
        'Amount': np.random.exponential(50, n),
        'Time_of_Day': np.random.randint(0, 24, n),
        'Distance_from_Home_km': np.random.normal(5, 3, n),
        'Merchant_Type': np.random.choice(['Grocery', 'Gas', 'Online', 'Retail'], n),
        'Fraud_Label': 0
    }
    df = pd.DataFrame(data)

    anomaly_idx = df.sample(frac=0.01).index
    df.loc[anomaly_idx, 'Amount'] *= 20
    df.loc[anomaly_idx, 'Distance_from_Home_km'] *= 50
    df.loc[anomaly_idx, 'Time_of_Day'] = np.random.choice([2, 3, 4], len(anomaly_idx))
    df.loc[anomaly_idx, 'Fraud_Label'] = 1

    df['Distance_from_Home_km'] = df['Distance_from_Home_km'].clip(lower=0)
    return df
