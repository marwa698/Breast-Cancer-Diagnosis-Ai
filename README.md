# 🏥 Breast Cancer Diagnosis AI

A multi-modal AI system for breast cancer decision support, combining clinical data analysis and medical image classification.

> ⚠️ **Disclaimer:** This tool is for research and educational purposes only. It is not a substitute for professional medical diagnosis.

---

## 🚀 Features

- **Dual-Model Architecture:** Tabular ML + Deep Learning CNN
- **XGBoost Classifier** on Wisconsin Breast Cancer Dataset — 97.4% accuracy
- **EfficientNet-B0** on Breast Ultrasound Images — 95.3% accuracy  
- **SHAP Explainability** — understand why the model made its decision
- **Grad-CAM Visualization** — see what the CNN focused on in the image
- **Combined Report** — both models cross-validate each other

---

## 🧠 Architecture

### System 1 — Tabular ML
| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| XGBoost ✅ | 97.4% | 0.994 |
| SVM | 97.4% | — |
| Logistic Regression | 96.5% | — |

### System 2 — Deep Learning
| Model | Val Accuracy | Dataset |
|-------|-------------|---------|
| EfficientNet-B0 ✅ | 95.3% | BUSI (1,578 images) |

---

## 📁 Project Structure
breast-cancer-diagnosis-ai/

├── app/

│   ├── main.py                  # Streamlit entry point

│   ├── utils.py

│   ├── tabular/

│   │   ├── preprocessing.py

│   │   ├── train_models.py

│   │   ├── predictor.py

│   │   └── explainability.py

│   ├── imaging/

│   │   ├── image_preprocessing.py

│   │   ├── cnn_model.py

│   │   ├── train_cnn.py

│   │   ├── image_predictor.py

│   │   └── gradcam.py

│   └── components/

│       ├── tabular_form_ui.py

│       ├── image_upload_ui.py

│       ├── results_panel.py

│       └── combined_report.py

├── data/

├── models/

├── notebooks/

└── tests/
---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-diagnosis-ai.git
cd breast-cancer-diagnosis-ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
streamlit run app/main.py
```

---

## 📊 Datasets

- **Tabular:** [Wisconsin Breast Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) — 569 samples, 30 features
- **Images:** [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) — 1,578 images

---

## 🛠️ Tech Stack

`Python` `Streamlit` `XGBoost` `PyTorch` `EfficientNet-B0` `SHAP` `Grad-CAM` `scikit-learn` `OpenCV`

---

## 👩‍💻 Author

Made with ❤️ for portfolio purposes.