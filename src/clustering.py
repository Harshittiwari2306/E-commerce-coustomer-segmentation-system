import joblib
import os

def load_model():
    base_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_path, "models", "kmeans_model.pkl")
    
    return joblib.load(model_path)