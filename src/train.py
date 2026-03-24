import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from scipy.sparse import hstack


def main():
    # -----------------------------
    # 1. Load dataset (SAFE READ)
    # -----------------------------
    df = pd.read_csv(
        "data/raw/creditcard.csv",
        low_memory=False
    )

    # -----------------------------
    # 2. Create dummy text column (since creditcard.csv has no text)
    # -----------------------------
    df["description"] = "normal transaction"

    # -----------------------------
    # 3. Features & target
    # -----------------------------
    y = df["Class"]
    X_num = df.drop(columns=["Class", "description"])

    # -----------------------------
    # 4. Train-test split
    # -----------------------------
    X_train_num, X_test_num, y_train, y_test = train_test_split(
        X_num,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # 5. Scale numerical features
    # -----------------------------
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    # -----------------------------
    # 6. TF-IDF on text
    # -----------------------------
    tfidf = TfidfVectorizer(
        max_features=10,
        stop_words="english"
    )

    X_train_text = tfidf.fit_transform(
        df.loc[X_train_num.index, "description"]
    )
    X_test_text = tfidf.transform(
        df.loc[X_test_num.index, "description"]
    )

    # -----------------------------
    # 7. Combine numerical + text
    # -----------------------------
    X_train_final = hstack([X_train_num_scaled, X_train_text])
    X_test_final = hstack([X_test_num_scaled, X_test_text])

    # -----------------------------
    # 8. Train Random Forest (LIGHT & STABLE)
    # -----------------------------
    rf = RandomForestClassifier(
        n_estimators=30,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )

    rf.fit(X_train_final, y_train)

    # -----------------------------
    # 9. Evaluation
    # -----------------------------
    y_pred = rf.predict(X_test_final)
    y_prob = rf.predict_proba(X_test_final)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC:", roc_auc)

    # -----------------------------
    # 10. Save model artifacts
    # -----------------------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(rf, "models/fraud_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(tfidf, "models/tfidf.pkl")

    print("\nModels saved successfully.")


if __name__ == "__main__":
    main()
