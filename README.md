 # 💳 Credit Card Fraud Detection

Detecting fraudulent credit card transactions using machine learning techniques.  
This project builds a complete end-to-end pipeline — from data preprocessing and feature engineering to model training, evaluation, and prediction.

---

## 📘 Overview
Credit card fraud is a major issue in financial systems, where fraudulent transactions are rare but costly.  
This project demonstrates how to use machine learning to identify such anomalies in highly imbalanced datasets with high precision and recall.

The notebook and scripts showcase data exploration, preprocessing, model building, evaluation, and performance visualization.

---

## 🚀 Features
- Clean and reproducible ML workflow  
- Handles data imbalance using SMOTE / class weighting  
- Multiple models: Logistic Regression, Random Forest, XGBoost  
- Evaluation using ROC-AUC, Precision-Recall, F1-score  
- Model persistence with `joblib`  
- Optional SHAP-based explainability  
- Ready-to-run Jupyter notebook for experimentation  

````

---

## 🧠 Dataset
The project uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), which contains transactions made by European cardholders in September 2013.

- **Total Transactions:** 284,807  
- **Fraudulent Transactions:** 492 (0.172%)  
- **Features:** 30 (V1–V28 PCA components + Time + Amount)  
- **Target:** `Class` → 1 (Fraud), 0 (Legit)

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt
````

---

## 🧩 Usage

### 🏋️‍♂️ Train the Model

```bash
python src/train.py
```

### 🧪 Evaluate

```bash
python src/evaluate.py
```

### 🔍 Predict on New Transactions

```bash
python src/predict.py --input sample.csv
```

Results and reports will be saved under `/reports`.

---

## 📊 Model Performance (Example)

|   Metric  | Logistic Regression | Random Forest | XGBoost |
| :-------: | :-----------------: | :-----------: | :-----: |
|  Accuracy |        99.93%       |     99.95%    |  99.96% |
| Precision |         0.88        |      0.92     |   0.94  |
|   Recall  |         0.82        |      0.86     |   0.89  |
|  ROC-AUC  |         0.99        |     0.998     |  0.999  |

*(Values are illustrative — replace with your real results.)*

---

## 🧭 Future Improvements

* Incorporate deep learning (Autoencoders / LSTMs)
* Real-time detection API using FastAPI or Flask
* Feature importance visualization via SHAP
* Model deployment with Streamlit dashboard

---

## 🧰 Tech Stack

* **Language:** Python 3.9+
* **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn, XGBoost, imbalanced-learn, SHAP

---

### Connect with me:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-its--vishnu--verma-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/its-vishnu-verma/)
[![Email](https://img.shields.io/badge/Email-ui22ec86@iiitsurat.ac.in-D14836?style=flat-square&logo=gmail)](mailto:ui22ec86@iiitsurat.ac.in)
