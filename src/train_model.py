import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Base path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load data
data_path = os.path.join(base_path, "data", "smartcart_customers.csv")
print("Loading from:", data_path)

df = pd.read_csv(data_path)

# -------------------------------
# 🔥 Handle missing values
# -------------------------------
# Keep only numeric columns
X = df.select_dtypes(include=['int64', 'float64'])

# Fill NaN with median (BEST practice)
X = X.fillna(X.median())

# -------------------------------
# Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Train model
# -------------------------------
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# -------------------------------
# Save model
# -------------------------------
os.makedirs(os.path.join(base_path, "models"), exist_ok=True)

joblib.dump(kmeans, os.path.join(base_path, "models", "kmeans_model.pkl"))
joblib.dump(scaler, os.path.join(base_path, "models", "scaler.pkl"))

print("✅ Model saved successfully")