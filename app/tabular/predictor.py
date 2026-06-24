import joblib
import numpy as np
from app.tabular.preprocessing import preprocess_input, validate_input

def load_model(model_path: str = "models/tabular_model.pkl"):
    return joblib.load(model_path)

def predict(input_dict: dict):
    is_valid, msg = validate_input(input_dict)
    if not is_valid:
        raise ValueError(msg)

    X = preprocess_input(input_dict)
    model = load_model()
    
    prediction = int(model.predict(X)[0])
    probability = model.predict_proba(X)[0]
    
    # تحويل صريح لـ float
    prob_benign = float(probability[0]) * 100
    prob_malignant = float(probability[1]) * 100
    confidence = prob_benign if prediction == 0 else prob_malignant

    label = "Malignant 🔴" if prediction == 1 else "Benign 🟢"

    return {
        "prediction": prediction,
        "label": label,
        "confidence": round(confidence, 2),
        "prob_benign": round(prob_benign, 2),
        "prob_malignant": round(prob_malignant, 2)
    }
    