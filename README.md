# pepper-disease-app
# 🌿 Black Pepper Disease Prediction App

This is a Streamlit web application that predicts the occurrence of major black pepper diseases based on weather parameters. The app uses a machine learning model trained on three years of historical data and provides disease occurrence predictions based on inputs such as temperature, humidity, sunshine, and rainfall.

## 🚀 Live Demo
👉 [Add your Streamlit Cloud link here after deployment]

## 📌 Features
- Input forecasted weather parameters
- Predict the likelihood of disease occurrence (yes/no)
- User-friendly interface powered by Streamlit
- Built using Logistic Regression for reliable and balanced predictions

## 🧠 ML Model Details
- Model: Logistic Regression (best performing model based on F1 score)
- Target: Occurrence of black pepper diseases such as algal rust, pollu disease, slow wilt, and quick wilt
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score

## 📂 Files Included
- `pepper_app.py`: Streamlit app source code
- `black_pepper_rf_model.pkl`: Trained machine learning model
- `requirements.txt`: Python dependencies for running the app

## ▶️ How to Run Locally

1. Clone the repository or download the files
2. Install dependencies:

```bash
pip install -r requirements.txt
