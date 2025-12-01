import streamlit as st
import pandas as pd
import joblib

from src.data_loader import load_data
from src.preprocessing import build_feature_matrix
from src.config import MODELS_DIR


# ---------- CACHED HELPERS ----------

@st.cache_data
def get_feature_stats():
    """
    Load dataset, build feature matrix, and compute column means.
    This ensures new prediction rows have all required columns.
    """
    df = load_data()
    X, y = build_feature_matrix(df)
    col_means = X.mean()
    return X.columns.tolist(), col_means


@st.cache_resource
def load_best_model():
    """
    Load the best trained model (RandomForest).
    If later you change best model, update the filename here.
    """
    model_path = MODELS_DIR / "RandomForest_model.joblib"
    model = joblib.load(model_path)
    return model


# ---------- STREAMLIT UI ----------

st.set_page_config(page_title="Iron Ore Quality Prediction", layout="centered")

st.title("🧪 Iron Ore Silica Concentrate Prediction")
st.write(
    """
    This app uses a trained **RandomForest** model to predict 
    **% Silica Concentrate** based on key flotation process variables.
    """
)

# Load model and feature stats
all_features, col_means = get_feature_stats()
model = load_best_model()

st.sidebar.header("Input Process Parameters")

# Choose a set of important features for user input
IMPORTANT_FEATURES = [
    "% Silica Feed",
    "% Iron Feed",
    "% Iron Concentrate",
    "Starch Flow",
    "Amina Flow",
    "Ore Pulp Density",
    "Ore Pulp pH",
    "Flotation Column 01 Air Flow",
    "Flotation Column 02 Air Flow",
    "Flotation Column 03 Air Flow",
    "Flotation Column 01 Level",
    "Flotation Column 02 Level",
    "hour",
    "dayofweek",
    "month",
]

user_inputs = {}

for feat in IMPORTANT_FEATURES:
    if feat not in col_means.index:
        st.sidebar.warning(f"Feature '{feat}' not found in training data. Skipping.")
        continue

    default_value = float(col_means[feat])

    if feat in ["hour"]:
        user_inputs[feat] = st.sidebar.slider("Hour of day", 0, 23, int(default_value))
    elif feat in ["dayofweek"]:
        user_inputs[feat] = st.sidebar.slider("Day of week (0=Mon)", 0, 6, int(default_value))
    elif feat in ["month"]:
        user_inputs[feat] = st.sidebar.slider("Month", 1, 12, int(default_value))
    else:
        # numerical features
        user_inputs[feat] = st.sidebar.number_input(
            feat,
            value=default_value,
            format="%.4f"
        )

if st.sidebar.button("Predict % Silica Concentrate"):
    # Start from mean values of all features
    row = col_means.copy()

    # Override with user inputs
    for feat, val in user_inputs.items():
        if feat in row.index:
            row[feat] = val

    # Create single-row DataFrame with all features
    X_new = pd.DataFrame([row])

    # Make prediction
    y_pred = model.predict(X_new)[0]

    st.subheader("Predicted % Silica Concentrate")
    st.success(f"{y_pred:.4f}")

    st.write(
        """
        **Note:** This prediction is based on a RandomForest model trained on historical
        flotation plant data. It should be used as a **decision-support tool**,
        not as a hard replacement for laboratory analysis.
        """
    )
else:
    st.info("Set parameters in the sidebar and click **Predict % Silica Concentrate**.")
