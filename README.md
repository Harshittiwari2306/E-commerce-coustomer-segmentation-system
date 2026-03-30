# 🛒 SmartCart: E-commerce Customer Segmentation System

## 📌 Overview

SmartCart is an end-to-end machine learning project that segments e-commerce customers into meaningful groups using **K-Means clustering**. The goal is to uncover customer behavior patterns and provide actionable business insights for targeted marketing strategies.

This project goes beyond a simple model by implementing a **complete ML pipeline + interactive Streamlit dashboard**.

---

## 🚀 Features

* Data Cleaning & Preprocessing
* Feature Engineering (Total Spending, Customer Tenure, etc.)
* Handling Missing Values (Median Imputation)
* Feature Scaling using StandardScaler
* Optimal Cluster Selection (Elbow Method + KneeLocator)
* K-Means Clustering
* Customer Segment Profiling
* Interactive Streamlit Dashboard

---

## 📊 Customer Segments

| Cluster | Segment Name        | Description                            |
| ------- | ------------------- | -------------------------------------- |
| 0       | Average Customers   | Moderate income and spending behavior  |
| 1       | Premium Customers   | High income, high spending, high value |
| 2       | Low-Value Customers | Low engagement and minimal spending    |
| 3       | Loyal Customers     | Frequent buyers with long tenure       |

---

## 🧠 Tech Stack

* **Programming:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit
* **Model Saving:** Joblib

---

## 📂 Project Structure

```
SmartCart-Customer-Segmentation/
│
├── app/
│   └── app.py
│
├── data/
│   └── smartcart_customers.csv
│
├── models/
│   ├── kmeans_model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── train_model.py
│   ├── preprocessing.py
│   └── utils.py
│
├── notebooks/
│   └── smartcart.ipynb
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/SmartCart-Customer-Segmentation.git
cd SmartCart-Customer-Segmentation
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### 1️⃣ Train the Model

```bash
python src/train_model.py
```

### 2️⃣ Run Streamlit App

```bash
streamlit run app/app.py
```

---

## 📈 Business Insights

* 💰 Premium customers contribute the highest revenue → focus on retention
* 🔁 Loyal customers show repeat purchases → upsell opportunities
* 📊 Average customers can be converted with targeted discounts
* ⚠️ Low-value customers require re-engagement strategies

---

## 🧠 Key Learnings

* Importance of data preprocessing in ML pipelines
* Handling missing values effectively
* Feature scaling for clustering algorithms
* Translating ML outputs into business insights
* Building interactive dashboards using Streamlit

---

## 🔮 Future Improvements

* Deploy application on cloud (Streamlit Cloud / AWS / Render)
* Add real-time customer prediction form
* Integrate recommendation system
* Use advanced clustering (DBSCAN / Hierarchical Clustering)

---

## 👤 Author

**Harshit Tiwari**
Aspiring Data Scientist | Machine Learning Enthusiast

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to connect!
