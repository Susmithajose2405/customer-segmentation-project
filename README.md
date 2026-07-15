# 🛍️ Smart Customer Segmentation System

A Streamlit web app that segments mall customers into meaningful groups using **K-Means clustering**, so businesses can design targeted marketing strategies for each group.

---

## 📁 Project Structure

```
customer-segmentation-system/
│
├── app.py                          # Main Streamlit App
├── Mall_Customers.csv              # Sample dataset
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── assets/
│   └── style.css                   # Custom UI styling
│
├── images/
│   └── logo.png                    # App logo
│
└── output/
    └── Customer_Segmentation_Output.csv   # Generated after clustering
```

---

## 🚀 Features

- **Flexible data input** — use the built-in sample dataset or upload your own CSV
- **Interactive EDA** — distributions, correlation heatmap, gender split, scatter plots
- **Optimal cluster detection** — Elbow Method + Silhouette Score chart to help pick the best `k`
- **K-Means clustering** — with adjustable number of clusters via a slider
- **2D visualization** — PCA-reduced scatter plot of segments, plus direct feature scatter with centroids
- **Segment insights** — per-cluster averages, cluster sizes, and auto-generated "High/Low/Avg" business labels
- **Downloadable output** — export the fully labeled dataset as CSV

---

## 🧠 How It Works (Pipeline)

1. **Load data** → CSV with numeric customer attributes (e.g., Age, Annual Income, Spending Score)
2. **Select features** → choose 2+ numeric columns to cluster on
3. **Scale features** → `StandardScaler` normalizes features so no single column dominates distance calculations
4. **Find optimal k** → WCSS (elbow) + silhouette score computed for a range of `k` values
5. **Run K-Means** → assigns every customer to a cluster
6. **Reduce dimensions** → PCA compresses features to 2D for visualization when more than 2 features are used
7. **Profile segments** → mean feature values per cluster + human-readable High/Low labels
8. **Export** → download the segmented CSV for use in CRM / marketing tools

---

## 🛠️ Setup & Installation

### 1. Clone / download this project
```bash
cd customer-segmentation-system
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 📊 About the Dataset

`Mall_Customers.csv` included here is a **synthetic sample** built to resemble the well-known "Mall Customer Segmentation" dataset structure:

| Column | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Gender | Male / Female |
| Age | Customer age |
| Annual Income (k$) | Yearly income in thousands |
| Spending Score (1-100) | Score assigned based on spending behavior |

> 💡 You can replace this file with the real Kaggle "Mall Customers" dataset, or upload any CSV with numeric columns — the app adapts automatically to whatever numeric features you select.

---

## 🎯 Typical Customer Segments You'll Discover

| Segment | Description | Suggested Strategy |
|---|---|---|
| High Income, High Spending | Premium/target customers | VIP loyalty programs, early access to new products |
| High Income, Low Spending | Careful spenders with money to spend | Personalized offers, trust-building campaigns |
| Low Income, High Spending | Impulsive/careless spenders | Budget-friendly bundles, installment options |
| Low Income, Low Spending | Price-sensitive customers | Discounts, value deals |
| Average Income, Average Spending | Standard/general customers | Regular promotions, retention campaigns |

*(Actual labels depend on your data and chosen `k`.)*

---

## 🔧 Tech Stack

- **Streamlit** — web app framework
- **scikit-learn** — K-Means, PCA, StandardScaler, silhouette score
- **pandas / numpy** — data handling
- **matplotlib / seaborn** — visualizations

---

## 📈 Possible Extensions

- Add **hierarchical clustering** or **DBSCAN** as alternative algorithms
- Add **3D cluster visualization** with Plotly
- Integrate a **recommendation engine** per segment
- Deploy on **Streamlit Community Cloud**, **Render**, or **HuggingFace Spaces**
- Add authentication for multi-user marketing team access

---

## 📄 License

Free to use and modify for learning and portfolio projects.