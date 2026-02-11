# CardioRisk – Heart Disease Risk Prediction System

CardioRisk is an end-to-end Machine Learning web application that predicts the **risk of heart disease** based on clinical parameters.  
The project includes data preprocessing, model training, evaluation, and deployment using **Streamlit**.

---

## 📌 Problem Statement
Early identification of heart disease risk can help in timely medical consultation.  
This project aims to build a **machine learning–based risk prediction system** using commonly available clinical features.

---

## 🧠 Features Used
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar
- Resting ECG Results
- Maximum Heart Rate Achieved
- Exercise-Induced Angina
- ST Depression (Oldpeak)

---

## ⚙️ Tech Stack
- **Python**
- **Pandas, NumPy** – data handling
- **Scikit-learn** – Random Forest model
- **Streamlit** – web application
- **Git & GitHub** – version control

---

## 🤖 Machine Learning Model
- **Algorithm:** Random Forest Classifier  
- **Evaluation Metrics:**
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
  - ROC-AUC Score

## 📊 Model Performance

```✅ Accuracy
Accuracy: 0.8369 (83.7%)
```

### 📈 Classification Report
## 📈 Classification Report

| Class        | Precision | Recall | F1-Score | Support |
|--------------|----------|--------|----------|---------|
| 0 (No Risk)  | 0.82     | 0.80   | 0.81     | 82      |
| 1 (Risk)     | 0.85     | 0.86   | 0.85     | 102     |


ROC-AUC Score: 0.9208

A ROC-AUC score above 0.90 indicates excellent class separability.

### 🔍 Confusion Matrix
```
[[66 16]
 [14 88]]
```

### Interpretation:

True Negatives (66): Correctly predicted no-risk cases

False Positives (16): Predicted risk but actually no risk

False Negatives (14): Missed actual risk cases

True Positives (88): Correctly predicted risk cases

The model maintains a strong balance between sensitivity (recall) and precision.

The model achieved strong performance with an ROC-AUC score above **0.90**, indicating good class separation.

---

## 🚀 How to Run Locally

### 1️⃣ Clone the repository
```bash

git clone https://github.com/<your-username>/CardioRisk.git
cd CardioRisk
```
### 2️⃣ Create and activate virtual environment
```bash
python -m venv ml_proj
ml_proj\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Train the model
```bash
python train_model.py
```
### 4️⃣ Train the model
```bash
streamlit run app.py
```
## 🌐 Deployment

The application is deployed using Streamlit Cloud and can be accessed via a public URL.

 https://cardiorisk-heart-disease-prediction-system-bbfrwawjctpsnoexcyn.streamlit.app/

##📂 Project Structure
```
CardioRisk/
├── app.py
├── train_model.py
├── data_preprocess.ipynb
├── heart_clean.csv
├── requirements.txt
├── README.md
├── .gitignore
└── model/
    └── heart_model.pkl
```


