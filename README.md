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
CardioRisk/
├── app.py # Streamlit web application
├── train_model.py # Model training and evaluation script
├── data_preprocess.ipynb # EDA and data preprocessing
├── heart_clean.csv # Cleaned dataset
├── requirements.txt # Project dependencies
├── README.md # Project documentation
├── .gitignore # Ignored files and folders
└── model/
└── heart_model.pkl # Trained Random Forest model

