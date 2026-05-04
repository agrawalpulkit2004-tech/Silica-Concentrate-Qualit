# Quality Prediction of Iron Ore Concentrate using Machine Learning

🎓 **Master Thesis Project — Mining Engineering**


---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Why This Matters](#-why-this-project-matters-in-mining-engineering)
- [Key Features](#-key-features)
- [Installation](#-installation-guide)
- [Usage](#-usage)
- [Model Performance](#-model-performance-summary)
- [Industrial Applications](#-industrial-relevance--applications)
- [Limitations & Future Scope](#-limitations--future-scope)
- [Citation](#-citation)
- [Contact](#-author--contact)

---

## 📌 Project Overview

This project develops a **Machine Learning-based Soft Sensor** for real-time prediction of **% Silica Concentrate** in iron ore using process parameters from a flotation-based beneficiation plant.

Traditional laboratory testing for silica impurity takes **3–6 hours**, making process control reactive instead of proactive. The developed ML model solves this by providing **instant virtual silica estimation** using plant data such as:

- Amine Flow
- Air Flow
- Pulp Density
- pH
- Iron Grade

### 🎯 Key Goals:

✔ Predict silica impurity in real-time  
✔ Optimize flotation control parameters  
✔ Enhance product quality, recovery, and plant yield  
✔ Enable Industry 4.0-based Smart Beneficiation

---

## ✨ Key Features

- 🤖 **Multiple ML Models**: Linear Regression, XGBoost, Random Forest
- 📊 **Advanced Explainability**: SHAP analysis for feature importance
- 🎯 **High Accuracy**: Random Forest achieves 94.6% R² score
- 🌐 **Interactive UI**: Streamlit-based web interface
- ⚙️ **Modular Architecture**: Well-organized, extensible codebase
- 📈 **Comprehensive Reporting**: Detailed metrics, visualizations, and comparisons
- 🔧 **Production-Ready**: Model serialization with joblib

---

## 🛠 Why This Project Matters in Mining Engineering

| Mining Challenge             | ML-Based Solution                 |
| ---------------------------- | --------------------------------- |
| 3–6 hrs delay in lab testing | Instant prediction of % Silica    |
| High reagent wastage         | Optimize chemical dosage          |
| Poor process visibility      | SHAP-based feature explainability |
| Inconsistent product grade   | Real-time quality monitoring      |
| No digital optimization      | SCADA/DCS integration-ready       |

---

## 🗂 Project Structure

```
iron-ore-quality-mtp/
├── data/
│   ├── raw/
│   │   └── MiningProcess_Flotation_Plant_Database.csv
│   └── processed/
├── models/
│   ├── LinearRegression_model.joblib
│   ├── RandomForest_model.joblib
│   ├── XGBoost_model.joblib
│   └── rf_silica_model.joblib
├── reports/
│   ├── figures/
│   └── results/
│       ├── feature_importance.csv
│       ├── model_comparison.json
│       └── predictions_vs_actual.csv
├── notebooks/
│   └── eda_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── run_training.py
├── run_evaluation.py
├── run_inference_example.py
├── app.py
├── rerun.py
├── requirements.txt
└── README.md
```

---

## 🖼 System Architecture (ML-Assisted Flotation Monitoring)

### Data Pipeline & ML Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FLOTATION PLANT SENSORS                              │
│   (Air Flow, Amine Flow, Pulp Density, pH, Iron Grade, Tailings Grade)     │
└────────────────────────┬────────────────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │    DATA COLLECTION & STORAGE       │
        │  (CSV → MiningProcess_Database)    │
        └────────────┬───────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │  DATA PREPROCESSING & CLEANING     │
        │  • Handle missing values           │
        │  • Remove outliers                 │
        │  • Normalize/Scale features        │
        └────────────┬───────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │   FEATURE ENGINEERING              │
        │  • Statistical features            │
        │  • Derived variables               │
        │  • Correlation analysis            │
        └────────────┬───────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │   MODEL TRAINING & SELECTION       │
        │  ┌─────────────────────────────┐  │
        │  │ • Linear Regression         │  │
        │  │ • XGBoost                   │  │
        │  │ • Random Forest ⭐ (BEST)  │  │
        │  └─────────────────────────────┘  │
        └────────────┬───────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────────┐  ┌─────────────────────┐
│ MODEL EVALUATION    │  │  EXPLAINABILITY     │
│ • R² Score          │  │  (SHAP Analysis)    │
│ • MAE/RMSE          │  │  • Feature Impact   │
│ • Cross-validation  │  │  • Prediction Force │
└─────────────────────┘  └─────────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │   REAL-TIME INFERENCE              │
        │  • Load trained model              │
        │  • Process new sensor data         │
        │  • Generate silica prediction      │
        └────────────┬───────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
    ┌─────────────┐         ┌──────────────┐
    │ STREAMLIT   │         │ REPORTS &    │
    │ WEB APP     │         │ DASHBOARDS   │
    └─────────────┘         └──────────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │   ACTIONABLE INSIGHTS              │
        │  ✔ Optimize Reagent Dosage        │
        │  ✔ Adjust Air Flow Parameters     │
        │  ✔ Real-time Quality Monitoring   │
        │  ✔ SCADA/DCS Integration Ready    │
        └────────────────────────────────────┘
```

### Pipeline Components Description

| Component               | Responsibility                          | Tools/Libraries                |
| ----------------------- | --------------------------------------- | ------------------------------ |
| **Data Collection**     | Acquire plant sensor data               | CSV, SCADA Systems             |
| **Preprocessing**       | Clean, normalize, handle missing values | Pandas, NumPy                  |
| **Feature Engineering** | Create meaningful features              | Scikit-learn, Pandas           |
| **Model Training**      | Train multiple ML models                | Scikit-learn, XGBoost          |
| **Model Selection**     | Choose best performer                   | Scikit-learn metrics           |
| **Explainability**      | Understand model decisions              | SHAP                           |
| **Inference**           | Make real-time predictions              | Trained models, Joblib         |
| **Visualization**       | Display results & insights              | Matplotlib, Seaborn, Streamlit |

---

## ⚙️ Installation Guide

### 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)
- At least 2GB free disk space

### 📌 Step 1: Clone Repository

```bash
git clone https://github.com/agrawalpulkit2004-tech/Silica-Concentrate-Qualit.git
cd iron-ore-quality-mtp
```

### 📌 Step 2: Create Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 📌 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📖 Usage

### 🚀 Quick Start

After installation, you can run the complete pipeline:

```bash
# Train models
python run_training.py

# Evaluate models
python run_evaluation.py

# Run inference on sample data
python run_inference_example.py

# Launch interactive web app
streamlit run app.py
```

---

## 🧠 Train Models

```bash
python run_training.py
```

**Features:**

- ✔ Trains Linear Regression, XGBoost, and Random Forest
- ✔ Selects best model based on R² score
- ✔ Saves models to `/models/`

---

## 📊 Model Evaluation

```bash
python run_evaluation.py
```

**Generates:**

| Output File                 | Description                    |
| --------------------------- | ------------------------------ |
| `feature_importance.csv`    | Ranked feature importance      |
| `predictions_vs_actual.csv` | Model accuracy visualization   |
| `model_comparison.json`     | Performance metrics comparison |
| SHAP plots                  | Explainability analysis        |

---

## 🔮 Run Inference on Sample Data

```bash
python run_inference_example.py
```

**Features:**

- ✔ Uses pre-trained best model
- ✔ Processes sample input data
- ✔ Generates silica concentration predictions
- ✔ Saves predictions to `/reports/results/`

---

## 🌐 Run Streamlit App

```bash
streamlit run app.py
```

**Features:**

- ➡ Upload process input data
- ➡ Predict % Silica Concentrate
- ➡ View feature influence & SHAP summary
- ➡ Interactive model inference

---

## 📈 Model Performance Summary

| Model             | R² Score     | Comments                             |
| ----------------- | ------------ | ------------------------------------ |
| Linear Regression | 0.688        | Poor nonlinear capability            |
| XGBoost           | 0.841        | Good, but slightly less stable       |
| **Random Forest** | **⭐ 0.946** | **Best accuracy & interpretability** |

---

## 🌍 Industrial Relevance & Applications

✔ Real-time flotation process monitoring  
✔ Soft-sensor for silica impurity control  
✔ Reduced lab testing delays (3–6 hrs → seconds)  
✔ Automatic reagent dosage optimization  
✔ Supports SCADA/DCS integration  
✔ Step toward Smart Plant Automation & Industry 4.0

---

## 📎 Limitations & Future Scope

### 🔴 Current Limitations:

- Dataset from single industrial source
- No mineralogical features (XRD/FTIR) included
- Limited to flotation beneficiation plants

### 🟢 Future Enhancements:

- SCADA integration for real-time plant deployment
- Include XRD/FTIR mineralogy data
- Real-time reinforcement learning for reagent control
- Multi-plant model generalization
- Edge computing deployment

---

## 🔧 Troubleshooting & Common Issues

| Issue                         | Solution                                                                        |
| ----------------------------- | ------------------------------------------------------------------------------- |
| `ModuleNotFoundError`         | Ensure virtual environment is activated and dependencies are installed          |
| `FileNotFoundError` for CSV   | Verify raw data exists in `data/raw/MiningProcess_Flotation_Plant_Database.csv` |
| Streamlit port already in use | Run `streamlit run app.py --server.port 8502`                                   |
| Model files not found         | Run `python run_training.py` to generate model files                            |
| SHAP plots not displaying     | Update matplotlib: `pip install --upgrade matplotlib`                           |

---

---

## 📄 License

This project is part of a Master's thesis. Usage for academic and research purposes is permitted with proper attribution.
