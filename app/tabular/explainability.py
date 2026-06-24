import shap
import numpy as np
import matplotlib.pyplot as plt
import joblib
import io
import base64

def get_shap_explanation(input_dict: dict, model_path: str = "models/tabular_model.pkl"):
    from app.tabular.preprocessing import preprocess_input, load_feature_names

    model = joblib.load(model_path)
    X = preprocess_input(input_dict)
    feature_names = load_feature_names()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # تحويل آمن لـ numpy
    import numpy as np
    sv = np.array(shap_values)
    if sv.ndim == 3:
        sv = sv[1][0]  # خد class 1 (malignant)
    elif sv.ndim == 2:
        sv = sv[0]
    
    vals = np.abs(sv)
    top_indices = np.argsort(vals)[::-1][:5]
    top_features = [(feature_names[i], round(float(sv[i]), 4)) 
                    for i in top_indices]

    return top_features

def plot_shap_bar(top_features: list):
    """
    ترسم bar chart للـ SHAP values وترجعه كـ base64 image
    """
    features = [f[0] for f in top_features]
    values = [f[1] for f in top_features]
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in values]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(features[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("SHAP Value (impact on prediction)")
    ax.set_title("Top 5 Feature Impact")
    ax.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_base64