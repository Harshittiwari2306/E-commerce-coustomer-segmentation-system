import streamlit as st
import pandas as pd
import sys
import os
import joblib

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_data
from src.utils import cluster_names

st.set_page_config(page_title="SmartCart AI", layout="wide")

st.title("🛒 SmartCart Customer Segmentation")

# -------------------------------
# Load model + scaler safely
# -------------------------------
@st.cache_resource
def load_artifacts():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    model_path = os.path.join(base_path, "models", "kmeans_model.pkl")
    scaler_path = os.path.join(base_path, "models", "scaler.pkl")

    # Debug prints
    print("Model Path:", model_path)
    print("Scaler Path:", scaler_path)
    print("Model Exists:", os.path.exists(model_path))
    print("Scaler Exists:", os.path.exists(scaler_path))

    if not os.path.exists(model_path):
        raise FileNotFoundError("kmeans_model.pkl not found")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError("scaler.pkl not found")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload your customer dataset", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📂 Raw Data")
        st.dataframe(df.head())

        # Preprocessing
        df_processed = preprocess_data(df)

        # Numeric features only
        features = df_processed.select_dtypes(include=['int64', 'float64'])

        # Feature alignment (prevents mismatch crash)
        if hasattr(model, "feature_names_in_"):
            missing_cols = set(model.feature_names_in_) - set(features.columns)
            for col in missing_cols:
                features[col] = 0
            features = features[model.feature_names_in_]

        # Scaling
        features_scaled = scaler.transform(features)

        # Prediction
        df['Cluster'] = model.predict(features_scaled)
        df['Segment'] = df['Cluster'].map(cluster_names)

        st.subheader("📊 Clustered Data")
        st.dataframe(df.head())

        # Charts
        st.subheader("📈 Cluster Distribution")
        st.bar_chart(df['Segment'].value_counts())

        if 'Total_Spending' in df.columns:
            st.subheader("💰 Average Spending by Segment")
            st.bar_chart(df.groupby('Segment')['Total_Spending'].mean())

        # Insights
        st.subheader("🧠 Business Insights")
        st.markdown("""
        - 🟢 Premium Customers → Focus on retention  
        - 🔵 Loyal Customers → Upsell strategies  
        - 🟡 Average Customers → Discounts  
        - 🔴 Low-Value Customers → Re-engagement  
        """)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("📂 Upload a CSV file to start analysis.")