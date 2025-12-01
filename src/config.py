from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_PATH = BASE_DIR / "data" / "raw" / "MiningProcess_Flotation_Plant_Database.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "reports" / "results"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
