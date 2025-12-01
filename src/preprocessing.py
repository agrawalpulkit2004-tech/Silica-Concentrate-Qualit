import pandas as pd
from sklearn.model_selection import train_test_split
from .config import TEST_SIZE, RANDOM_STATE

TARGET = "% Silica Concentrate"  # primary prediction target


def _base_preprocess(df: pd.DataFrame):
    """
    Common preprocessing used both for training and for the Streamlit app.
    - remove duplicates
    - drop rows with missing target
    - extract time features (hour, month, dayofweek)
    - drop original 'date' column
    """
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["hour"] = df["date"].dt.hour
        df["month"] = df["date"].dt.month
        df["dayofweek"] = df["date"].dt.dayofweek
        df = df.drop(columns=["date"], errors="ignore")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def preprocess_data(df: pd.DataFrame):
    """
    Used during training:
    returns X_train, X_test, y_train, y_test.
    """
    X, y = _base_preprocess(df)
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


def build_feature_matrix(df: pd.DataFrame):
    """
    Used by Streamlit app:
    returns the full feature matrix X and target y (no splitting).
    """
    X, y = _base_preprocess(df)
    return X, y
