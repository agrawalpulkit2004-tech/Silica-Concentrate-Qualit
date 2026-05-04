import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import RESULTS_DIR, FIGURES_DIR


def evaluate_from_csv():
    """
    Use existing predictions_vs_actual.csv to compute metrics.
    No model file is required
    """
    pred_file = RESULTS_DIR / "predictions_vs_actual.csv"

    if not pred_file.exists():
        raise FileNotFoundError(
            f"{pred_file} not found. Make sure predictions_vs_actual.csv exists in reports/results."
        )

    df = pd.read_csv(pred_file)
    y_true = df["Actual"].values
    y_pred = df["Predicted"].values

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    print("\n📊 Evaluation Based on Stored Predictions:")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    return y_true, y_pred


def plot_error_distribution(errors: np.ndarray):
    """
    Plot residual (error) distribution.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, edgecolor="black")
    plt.title("Error Distribution (Residuals)")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_distribution.png", dpi=300)
    plt.close()


def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Plot Predicted vs Actual scatter with y = x reference line.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.title("Predicted vs Actual % Silica Concentrate")
    plt.xlabel("Actual % Silica Concentrate")
    plt.ylabel("Predicted % Silica Concentrate")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "prediction_vs_actual.png", dpi=300)
    plt.close()


def plot_feature_importance_from_csv():
    """
    Regenerate feature_importance_bar.png from feature_importance.csv
    with proper margins so labels are fully visible (no cropping).
    """
    fi_file = RESULTS_DIR / "feature_importance.csv"
    if not fi_file.exists():
        print(f"⚠️ {fi_file} not found. Skipping feature importance plot.")
        return

    df = pd.read_csv(fi_file)

    if not {"Feature", "Importance"}.issubset(df.columns):
        print("⚠️ feature_importance.csv does not contain 'Feature' and 'Importance' columns.")
        return

    # Sort and take top 15
    df_sorted = df.sort_values("Importance", ascending=False).head(15)
    # Reverse for top at top in barh
    df_sorted = df_sorted.iloc[::-1]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.barh(df_sorted["Feature"], df_sorted["Importance"])
    plt.title("Top 15 Feature Importances (RandomForest)", fontsize=14)
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)

    # Give extra space on the left for long labels
    plt.gcf().subplots_adjust(left=0.35)

    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "feature_importance_bar.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("✅ Regenerated feature_importance_bar.png with full labels.")


def re_evaluate():
    print("\n📌 Running Evaluation Using Stored Results (no model file needed)...")

    y_true, y_pred = evaluate_from_csv()

    print("\n📈 Generating plots...")
    plot_error_distribution(y_true - y_pred)
    plot_prediction_vs_actual(y_true, y_pred)
    plot_feature_importance_from_csv()

    print("\n🎯 Evaluation Completed Successfully!")
    print(f"Figures saved in: {FIGURES_DIR}")
    print(f"Results used from: {RESULTS_DIR}")


if __name__ == "__main__":
    re_evaluate()
