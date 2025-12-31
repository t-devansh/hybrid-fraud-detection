# model/features.py
from sklearn.preprocessing import LabelEncoder

def encode_features(df):
    le = LabelEncoder()
    df['Merchant_Type_Enc'] = le.fit_transform(df['Merchant_Type'])
    return df, le
