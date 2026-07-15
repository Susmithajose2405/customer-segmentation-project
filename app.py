"""
Smart Customer Segmentation System
------------------------------------
A Streamlit app that segments mall customers into meaningful groups
using K-Means clustering, so marketing teams can target each group
with the right strategy.

Run with: streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Customer Segmentation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# LOAD CUSTOM CSS
# --------------------------------------------------------------------------
def load_css(path):
    if os.path.exists(path):
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/style.css")

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 6])
with col_logo:
    if os.path.exists("images/logo.png"):
        st.image("images/logo.png", width=90)
with col_title:
    st.title("Smart Customer Segmentation System")
    st.caption("Segment your customers into actionable groups using Machine Learning (K-Means Clustering)")

st.divider()

# --------------------------------------------------------------------------
# SIDEBAR - DATA INPUT & SETTINGS
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

data_source = st.sidebar.radio(
    "Choose data source",
    ["Use sample dataset (Mall_Customers.csv)", "Upload your own CSV"]
)

if data_source == "Upload your own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("👈 Upload a CSV file to begin, or switch to the sample dataset.")
        st.stop()
else:
    df = pd.read_csv("Mall_Customers.csv")

st.sidebar.success(f"Loaded {df.shape[0]} customers, {df.shape[1]} columns")

# --------------------------------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# Drop obvious ID columns from feature choices
id_like = [c for c in numeric_cols if "id" in c.lower()]
default_features = [c for c in numeric_cols if c not in id_like]

st.sidebar.subheader("🎯 Select Features for Clustering")
features = st.sidebar.multiselect(
    "Pick 2 or more numeric columns",
    options=numeric_cols,
    default=default_features[:2] if len(default_features) >= 2 else default_features
)

if len(features) < 2:
    st.warning("Please select at least 2 numeric features from the sidebar to run clustering.")
    st.stop()

# --------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------
tab_overview, tab_eda, tab_cluster, tab_insights = st.tabs(
    ["📄 Data Overview", "📊 EDA", "🧩 Clustering", "💡 Segment Insights"]
)

# --------------------------------------------------------------------------
# TAB 1: DATA OVERVIEW
# --------------------------------------------------------------------------
with tab_overview:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(15), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", df.shape[0])
    c2.metric("Total Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Selected Features", len(features))

    st.subheader("Statistical Summary")
    st.dataframe(df[features].describe().T, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 2: EDA
# --------------------------------------------------------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Distribution of each feature**")
        fig, axes = plt.subplots(len(features), 1, figsize=(5, 3 * len(features)))
        if len(features) == 1:
            axes = [axes]
        for ax, col in zip(axes, features):
            sns.histplot(df[col], kde=True, ax=ax, color="#6C63FF")
            ax.set_title(col)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("**Correlation Heatmap**")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

        if "Gender" in df.columns:
            st.markdown("**Gender Distribution**")
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            df["Gender"].value_counts().plot(kind="bar", color=["#6C63FF", "#FF6584"], ax=ax3)
            st.pyplot(fig3)

    if len(features) >= 2:
        st.markdown(f"**{features[0]} vs {features[1]}**")
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        sns.scatterplot(data=df, x=features[0], y=features[1], ax=ax4, color="#6C63FF")
        st.pyplot(fig4)

# --------------------------------------------------------------------------
# TAB 3: CLUSTERING
# --------------------------------------------------------------------------
with tab_cluster:
    st.subheader("K-Means Clustering")

    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Elbow Method + Silhouette Score ---
    st.markdown("### Step 1: Find the Optimal Number of Clusters")
    max_k = st.slider("Max clusters to test (for elbow/silhouette chart)", 3, 15, 10)

    wcss = []
    sil_scores = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    col1, col2 = st.columns(2)
    with col1:
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        ax5.plot(list(k_range), wcss, marker="o", color="#6C63FF")
        ax5.set_xlabel("Number of Clusters (k)")
        ax5.set_ylabel("WCSS (Inertia)")
        ax5.set_title("Elbow Method")
        st.pyplot(fig5)

    with col2:
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        ax6.plot(list(k_range), sil_scores, marker="o", color="#FF6584")
        ax6.set_xlabel("Number of Clusters (k)")
        ax6.set_ylabel("Silhouette Score")
        ax6.set_title("Silhouette Analysis")
        st.pyplot(fig6)

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    st.info(f"💡 Suggested optimal k (highest silhouette score): **{best_k}**")

    # --- Final Clustering ---
    st.markdown("### Step 2: Run Final Clustering")
    n_clusters = st.slider("Choose number of clusters (k)", 2, max_k, best_k)

    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    final_sil = silhouette_score(X_scaled, df["Cluster"])

    st.success(f"✅ Clustering complete with k = {n_clusters} | Silhouette Score: {final_sil:.3f}")

    # --- PCA Visualization (2D) ---
    st.markdown("### Cluster Visualization (2D via PCA)")
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df["PCA1"], df["PCA2"] = components[:, 0], components[:, 1]

    fig7, ax7 = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("husl", n_clusters)
    sns.scatterplot(
        data=df, x="PCA1", y="PCA2", hue="Cluster",
        palette=palette, s=80, ax=ax7, legend="full"
    )
    ax7.set_title("Customer Segments (PCA-reduced view)")
    st.pyplot(fig7)

    # If exactly 2 features chosen, show a direct (non-PCA) scatter too
    if len(features) == 2:
        st.markdown(f"### Cluster Visualization ({features[0]} vs {features[1]})")
        fig8, ax8 = plt.subplots(figsize=(9, 5))
        sns.scatterplot(
            data=df, x=features[0], y=features[1], hue="Cluster",
            palette=palette, s=80, ax=ax8, legend="full"
        )
        centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
        ax8.scatter(centers_original[:, 0], centers_original[:, 1],
                    marker="X", s=250, c="black", label="Centroids")
        ax8.legend()
        st.pyplot(fig8)

    st.session_state["clustered_df"] = df
    st.session_state["features"] = features
    st.session_state["n_clusters"] = n_clusters

# --------------------------------------------------------------------------
# TAB 4: SEGMENT INSIGHTS
# --------------------------------------------------------------------------
with tab_insights:
    if "clustered_df" not in st.session_state:
        st.warning("Run clustering in the '🧩 Clustering' tab first.")
    else:
        cdf = st.session_state["clustered_df"]
        feats = st.session_state["features"]
        k = st.session_state["n_clusters"]

        st.subheader("Segment Profiles")

        profile = cdf.groupby("Cluster")[feats].mean().round(1)
        profile["Customer Count"] = cdf["Cluster"].value_counts().sort_index()
        st.dataframe(profile, use_container_width=True)

        st.markdown("### Cluster Sizes")
        fig9, ax9 = plt.subplots(figsize=(6, 4))
        cdf["Cluster"].value_counts().sort_index().plot(
            kind="bar", color=sns.color_palette("husl", k), ax=ax9
        )
        ax9.set_xlabel("Cluster")
        ax9.set_ylabel("Number of Customers")
        st.pyplot(fig9)

        st.markdown("### Suggested Business Labels")
        st.caption("Auto-generated hints based on relative feature values — refine these based on domain knowledge.")

        labels = []
        for c in sorted(cdf["Cluster"].unique()):
            row = profile.loc[c]
            desc = []
            for f in feats:
                overall_mean = cdf[f].mean()
                if row[f] > overall_mean * 1.15:
                    desc.append(f"High {f}")
                elif row[f] < overall_mean * 0.85:
                    desc.append(f"Low {f}")
                else:
                    desc.append(f"Avg {f}")
            labels.append({"Cluster": c, "Profile": ", ".join(desc), "Count": int(row["Customer Count"])})

        st.dataframe(pd.DataFrame(labels), use_container_width=True)

        st.markdown("### 📥 Download Segmented Data")
        os.makedirs("output", exist_ok=True)
        output_path = "output/Customer_Segmentation_Output.csv"
        cdf.to_csv(output_path, index=False)

        with open(output_path, "rb") as f:
            st.download_button(
                label="Download Customer_Segmentation_Output.csv",
                data=f,
                file_name="Customer_Segmentation_Output.csv",
                mime="text/csv",
            )

st.divider()
st.caption("Built with Streamlit, scikit-learn & ❤️ — Smart Customer Segmentation System")