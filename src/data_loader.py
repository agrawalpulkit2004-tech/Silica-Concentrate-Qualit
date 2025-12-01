import pandas as pd
from .config import DATA_RAW_PATH

def load_data():
    print("Using dataset from:", DATA_RAW_PATH)  # 👈 Add this
    df = pd.read_csv(DATA_RAW_PATH, decimal=",", low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    return df
