# app.py
from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle
import os

app = Flask(__name__)

MODEL_PATH = "models/final_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "❌ No file uploaded!"

    file = request.files["file"]
    if file.filename == "":
        return "❌ No selected file!"

    df = pd.read_csv(file)
    preds = model.predict(df)
    probas = None

    try:
        probas = model.predict_proba(df)[:, 1]
        df["fraud_probability"] = probas
    except:
        df["fraud_probability"] = "N/A"

    df["prediction"] = preds
    fraud_count = (preds == 1).sum()

    os.makedirs("uploads", exist_ok=True)
    output_path = os.path.join("uploads", "results.csv")
    df.to_csv(output_path, index=False)

    total = len(df)
    fraud_percent = round((fraud_count / total) * 100, 2)

    return render_template("results.html",
                           total=total,
                           frauds=fraud_count,
                           percent=fraud_percent,
                           file_path=output_path)

@app.route("/download")
def download():
    return send_file("uploads/results.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
