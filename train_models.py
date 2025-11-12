# train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import pickle
import os

# Load dataset
print("📂 Loading dataset...")
data = pd.read_csv("creditcard.csv")

# Separate features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Standardize Amount and Time
scaler = StandardScaler()
X["Time"] = scaler.fit_transform(X["Time"].values.reshape(-1, 1))
X["Amount"] = scaler.fit_transform(X["Amount"].values.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Balance the data using SMOTE
print("🔄 Applying SMOTE balancing...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)
}

results = []
best_model = None
best_f1 = 0

print("\n⚙️ Training and comparing models...\n")

for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([name, acc, rec, f1])
    print(f"{name}: Accuracy={acc:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    if f1 > best_f1:
        best_model = model
        best_f1 = f1

# Convert to DataFrame for viewing
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Recall", "F1"])
print("\n📊 Model Comparison:\n", results_df)

# Save best model
os.makedirs("models", exist_ok=True)
with open("models/final_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\n🏆 Best Model Saved → models/final_model.pkl ({type(best_model).__name__})")

# Create sample input file
sample_input = X_test.head(5)
sample_input.to_csv("sample_input.csv", index=False)
print("\n📁 sample_input.csv created for web testing.")
