import json
from .data_loader import load_data
from .preprocessing import preprocess_data
from .modeling import get_all_models, evaluate_model, save_model, run_shap_analysis
from .config import RESULTS_DIR


def run_training():
    df = load_data()

    # Optional: Use only 30% of data for testing during development
    # df = df.sample(frac=0.3, random_state=42)

    X_train, X_test, y_train, y_test = preprocess_data(df)

    models = get_all_models()
    results = {}

    best_model_name = None
    best_r2 = -1

    for name, model in models.items():
        print(f"\n🔵 Training {name} ...")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        print(f"{name} Performance: {metrics}")

        save_model(model, name)

        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_model_name = name
            best_model = model

    with open(RESULTS_DIR / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n🏆 Best Model: {best_model_name} with R2 = {best_r2}")
    shap_path = run_shap_analysis(best_model, X_test, best_model_name)

    if shap_path:
        print(f"SHAP Summary Plot saved at: {shap_path}")
    else:
        print("SHAP analysis not available for this model.")
