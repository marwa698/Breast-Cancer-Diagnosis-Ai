import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

def load_scaler(scaler_path: str = "models/scaler.pkl"):
    return joblib.load(scaler_path)

def load_feature_names(path: str = "models/feature_names.json"):
    with open(path, 'r') as f:
        return json.load(f)

def preprocess_input(input_dict: dict, scaler_path: str = "models/scaler.pkl"):
    feature_names = load_feature_names()
    
    clean_dict = {}
    for k, v in input_dict.items():
        # إزالة أي brackets أو spaces
        v_str = str(v).strip().strip('[]').strip()
        try:
            clean_dict[k] = float(v_str)
        except:
            clean_dict[k] = 0.0
    
    df = pd.DataFrame([clean_dict])
    df = df[feature_names]
    
    scaler = load_scaler(scaler_path)
    scaled = scaler.transform(df)
    return scaled

def validate_input(input_dict: dict):
    """
    تتأكد إن كل الـ features موجودة
    """
    feature_names = load_feature_names()
    missing = [f for f in feature_names if f not in input_dict]
    if missing:
        return False, f"Missing features: {missing}"
    return True, "OK"