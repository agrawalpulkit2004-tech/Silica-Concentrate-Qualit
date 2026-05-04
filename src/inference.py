import pandas as pd
from .modeling import load_model

def predict_one(input_dict):
    df = pd.DataFrame([input_dict])
    model = load_model()
    return model.predict(df)[0]
import pandas as pd
import joblib
from .modeling import load_model, MODELS_DIR

def predict_one(input_dict):
    df = pd.DataFrame([input_dict])
    
    model = load_model()
    
    # 🔥 Align features with trained model
    model_features = model.get_booster().feature_names
    df = df.reindex(columns=model_features, fill_value=0)
    
    return model.predict(df)[0]