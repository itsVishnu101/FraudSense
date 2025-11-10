## 🧠 FraudSense — Credit Card Fraud Detection 

> **FraudSense** is a complete machine learning project that detects fraudulent credit card transactions using the **Kaggle 2013 European Cardholders dataset**.
> It compares **Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost**, evaluates them on **Accuracy**, **Recall**, and **F1-score**, and deploys the best model via a **Flask web app**.

---

### 📂 Project Structure

```bash
FraudSense/
├── app.py                    # Flask web app for predictions
├── train_models.py           # Model training & comparison script
├── FraudSense_Colab.ipynb    # Google Colab notebook (EDA + model comparison)
├── requirements.txt          # Required Python libraries
├── README.md                 # Documentation (this file)
├── templates/
│   ├── index.html            # Upload UI
│   └── results.html          # Results page
├── static/
│   └── css/style.css         # Modern responsive UI styling
├── models/
│   └── final_model.pkl       # Saved best model (after training)
└── uploads/                  # Uploaded + predicted CSVs
```

---

## 🚀 Features

✅ Compares 4 ML models
✅ Evaluates with **Accuracy**, **Recall**, and **F1-score**
✅ Balances classes using **SMOTE**
✅ Saves the **best model automatically**
✅ Provides **Flask Web App** for predictions
✅ Clean and **responsive web UI**
✅ Optional **Google Colab notebook** for easy training and visualization

---

## 📊 Dataset Overview (2013 Credit Card Fraud)

* Source: [Kaggle — Credit Card Fraud Detection (2013)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* Records: **284,807 transactions**
* Features: **Time, Amount, V1–V28 (PCA features)**
* Target: **Class** → `0` = Legit, `1` = Fraud

### Example Data Snapshot

| Time | V1      | V2      | V3     | ... | Amount | Class |
| ---- | ------- | ------- | ------ | --- | ------ | ----- |
| 0    | -1.3598 | -0.0728 | 2.5363 | ... | 149.62 | 0     |
| 1    | 1.1918  | 0.2662  | 0.1664 | ... | 2.69   | 0     |

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/itsVishnu101/FraudSense.git
cd FraudSense
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Place dataset

Download the Kaggle 2013 `creditcard.csv` and place it in the **project root**.

### 4️⃣ Train and compare models

```bash
python train_models.py
```

This script will:

* Compare Logistic Regression, Decision Tree, Random Forest, and XGBoost
* Print Accuracy, Recall, and F1 for each
* Save the **best model** as `models/final_model.pkl`
* Generate `sample_input.csv` for quick web testing

---

## 📈 Model Comparison Example (Colab Results)

| Model               | Accuracy   | Recall   | F1-score   |
| ------------------- | ---------- | -------- | ---------- |
| Logistic Regression | 0.9992     | 0.90     | 0.94       |
| Decision Tree       | 0.9985     | 0.86     | 0.91       |
| Random Forest       | 0.9994     | 0.92     | 0.95       |
| XGBoost             | **0.9995** | **0.93** | **0.96** ✅ |

*(values vary slightly on each run depending on random seed & SMOTE)*

**Best Model:** 🏆 **XGBoost**

---

## 💻 Run the Flask App

After training, start the web server:

```bash
python app.py
```

Then open in browser:

```
http://127.0.0.1:5000/
```

### 🧾 Upload CSV for prediction

* Upload a file with columns: `Time, V1, V2, …, V28, Amount`
* Get downloadable CSV with:

  * `prediction` (0 = Legit, 1 = Fraud)
  * `fraud_probability` (if model supports it)

---

## 🌐 Web UI Preview

### 🏠 Home Page

Upload your transaction CSV to analyze:

```html
+-----------------------------------------+
| FraudSense – Credit Card Fraud Detector |
| Upload a CSV [Choose File] [Predict]    |
+-----------------------------------------+
```

### 📊 Results Page

Displays summary metrics:

```
Total Transactions: 300
Predicted Frauds: 2 (0.67%)
[Download Results]
```

---

## 🧪 Google Colab Notebook

To experiment interactively, open the provided notebook:

**📘 [FraudSense_Colab.ipynb](./FraudSense_Colab.ipynb)**

This includes:

* Data exploration & visualization
* Model comparison
* Metrics chart (Accuracy / Recall / F1)
* Automatic saving of best model

```python
# Example snippet
from xgboost import XGBClassifier
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train_bal, y_train_bal)
```

---

## 🧩 Tech Stack

| Category          | Technologies                                           |
| ----------------- | ------------------------------------------------------ |
| **Languages**     | Python 3                                               |
| **Libraries**     | scikit-learn, xgboost, imbalanced-learn, pandas, numpy |
| **Web Framework** | Flask                                                  |
| **Frontend**      | HTML, CSS (responsive UI)                              |
| **Notebook Env**  | Google Colab / Jupyter                                 |

---

## 📦 Deployment (Optional)

To containerize & deploy with Docker:

```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

Then:

```bash
docker build -t fraudsense .
docker run -p 5000:5000 fraudsense
```

---

## 📜 License

MIT License © 2025 — Developed by **Vishnu Verma**

---

## ⭐ Contribute

Pull requests are welcome!
If you like this project, please ⭐ it on GitHub — it helps a lot!

---

## 📬 Contact

**Author:** Vishnu Verma
**Email:** (ui22ec86@iiitsurat.ac.in)
**GitHub:** [itsVishnu101](https://github.com/itsVishnu101)

