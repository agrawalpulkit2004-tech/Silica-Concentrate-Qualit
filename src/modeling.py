import joblib
import shap
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .config import MODELS_DIR, FIGURES_DIR, RESULTS_DIR
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings("ignore")


def get_all_models():
    """Returns optimized models that train fast on CPU."""
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=50,      # Reduced for speed
            max_depth=12,         # Prevent deep trees
            n_jobs=-1,            # Use all CPU cores
            random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,     # Fast yet accurate
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,           # Full CPU parallelization
            tree_method="hist",  # Explicit CPU optimized
            random_state=42
        )
    }


def evaluate_model(model, X_test, y_test):
    """Return model performance metrics."""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)   # Always returns MSE
    rmse = mse ** 0.5                          # Convert to RMSE manually
    r2 = r2_score(y_test, y_pred)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def save_model(model, name):
    """Save trained model to disk."""
    path = MODELS_DIR / f"{name}_model.joblib"
    joblib.dump(model, path)
    return str(path)


def run_shap_analysis(model, X_test, model_name):
    """Generate SHAP explanation for best model."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:200])  # Use small subset for speed

        shap.summary_plot(shap_values, X_test[:200], show=False)
        shap_fig = FIGURES_DIR / f"{model_name}_shap_summary.png"
        plt.savefig(shap_fig, dpi=300, bbox_inches='tight')
        plt.close()
        return shap_fig
    except Exception as e:
        print(f"SHAP Analysis skipped for {model_name}: {e}")
        return None
