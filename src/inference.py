import pandas as pd
from .modeling import load_model

def predict_one(input_dict):
    df = pd.DataFrame([input_dict])
    model = load_model()
    return model.predict(df)[0]
