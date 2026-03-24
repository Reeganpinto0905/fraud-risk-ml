import joblib
import numpy as np
from scipy.sparse import hstack
FRAUD_THRESHOLD = 0.3  # business-defined threshold


def predict_transaction(num_features, text):
    model = joblib.load("models/fraud_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    tfidf = joblib.load("models/tfidf.pkl")

    num_scaled = scaler.transform([num_features])
    text_vec = tfidf.transform([text])

    final_input = hstack([num_scaled, text_vec])

    prob = model.predict_proba(final_input)[0][1]
    return prob

def simulate_transaction(num_features, text):
    print("\n📥 New Transaction Received")
    print("Description:", text)

    risk = predict_transaction(num_features, text)

    print(f"🔍 Fraud Risk Score: {risk:.2%}")

    return risk

def fraud_alert(risk_score):
    if risk_score >= FRAUD_THRESHOLD:
        print("🚨 FRAUD ALERT: Transaction flagged for review!")
    else:
        print("✅ Transaction appears safe.")




# Example
if __name__ == "__main__":
    example_num = [0] * 30
    example_text = "international transaction from suspicious location"

    risk = simulate_transaction(example_num, example_text)
    fraud_alert(risk)

