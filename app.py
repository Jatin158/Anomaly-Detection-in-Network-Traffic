from flask import Flask, request, render_template
import numpy as np
import dill as pickle
import os

app = Flask(__name__)

# Load models
model_paths = {
    "Random Forest": "random_forest_mimic.pkl",
    "XGBoost": "xgboost_mimic.pkl",
    "LightGBM": "lightgbm_mimic.pkl",
    "CatBoost": "catboost_mimic.pkl"
}

models = {}
for name, path in model_paths.items():
    with open(os.path.join(os.getcwd(), path), "rb") as f:
        models[name] = pickle.load(f)

# Class mapping
target_mapping = {
    0: 'Benign', 1: 'Brute Force', 2: 'DDoS-ACK_Fragmentation', 3: 'DDoS-HTTP_Flood',
    4: 'DDoS-ICMP_Flood', 5: 'DDoS-ICMP_Fragmentation', 6: 'DNS_Spoofing',
    7: 'DoS-HTTP_Flood', 8: 'DoS-SYN_Flood', 9: 'DoS-UDP_Flood',
    10: 'MITM-ArpSpoofing', 11: 'Sqlinjection', 12: 'Uploading_Attack',
    13: 'VulnerabilityScan', 14: 'XSS'
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        selected_model = request.form["model"]
        user_input = request.form["features"]
        feature_values = list(map(float, user_input.strip().split(",")))

        if len(feature_values) != 30:
            return render_template("result.html", attack="❌ Error: Enter exactly 30 values.")

        input_features = np.array(feature_values).reshape(1, -1)

        model = models[selected_model]
        probabilities = model.predict_proba(input_features)[0]
        max_index = np.argmax(probabilities)
        predicted_attack = target_mapping[max_index]

        return render_template("result.html", attack=f"{selected_model} predicts: {predicted_attack}")

    except Exception as e:
        return render_template("result.html", attack=f"❌ Error: {str(e)}")

# ⚠️ DO NOT include app.run() – Render uses Gunicorn to run the app!
