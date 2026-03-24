import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(df, target_col="Class", text_col="description", test_size=0.2):
    """
    Splits data into train and test sets.
    """
    X_num = df.drop(columns=[target_col, text_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X_num,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scales numerical features using StandardScaler.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
