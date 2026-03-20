# 🏥 Cancer Care AI - Detection System

An AI-powered web application that predicts breast cancer (Benign or Malignant) based on 30 tumor features using the **Wisconsin Breast Cancer Dataset**.

## 🚀 How to Run

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Model Training
Generate the machine learning model artifacts (first time only):
```bash
python train_model.py
```

### 3. Start the Application
Run the Flask server:
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## 🔬 Technical Details
- **Architecture**: Flask (Backend) + Vanilla JS/CSS (Frontend)
- **Model**: Random Forest Classifier (200 estimators)
- **Dataset**: UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)
- **Accuracy**: ~97.4% (Stratified Test Split)

## 🎨 Features
- **Instant Analysis**: Real-time prediction with confidence scores.
- **Glassmorphism UI**: Modern, premium dark mode interface.
- **Sample Loaders**: Quickly test with real dataset examples.
- **Responsive Design**: Optimized for desktops and mobile devices.

---
> ⚠️ **Disclaimer:** This tool is for educational purposes only and should not be used for actual medical diagnosis.
