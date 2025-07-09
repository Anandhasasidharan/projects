import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_features(df, drop_cols=['label', 'attack_cat', 'severity_score']):
    X = df.drop(columns=drop_cols)
    X_encoded = pd.get_dummies(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    return X_scaled, X_encoded.columns
