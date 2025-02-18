import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
import requests
from io import StringIO
from wordcloud import WordCloud

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import fetch_kddcup99

# ------------------------------
# Additional Model: XGBoost
# ------------------------------
import xgboost as xgb

# ------------------------------
# PyTorch imports for Autoencoder
# ------------------------------
import torch
import torch.nn as nn
import torch.optim as optim

# ==============================
# GLOBAL CONFIGURATION & STYLING
# ==============================
sns.set_theme(style="whitegrid")
st.set_page_config(
    page_title="Network Anomaly Detector Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# DATA LOADING MODULE
# ==============================
@st.cache_data(show_spinner=True)
def load_csv_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

@st.cache_data(show_spinner=True)
def load_url_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except Exception as e:
        st.error(f"Error loading data from URL: {e}")
        return None

@st.cache_data(show_spinner=True)
def load_kddcup_data():
    st.info("Loading sample KDD Cup 99 data...")
    try:
        data = fetch_kddcup99(percent10=True, subset='SA', random_state=42)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['true_label'] = [label.decode('utf-8') for label in data.target]
        return df
    except Exception as e:
        st.error(f"Error loading KDD Cup 99 data: {e}")
        return None

def get_data():
    st.sidebar.markdown("### Data Source")
    source = st.sidebar.radio("Select", ("Upload File", "URL", "Sample Data"))
    if source == "Upload File":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            df = load_csv_data(uploaded_file)
            if df is not None:
                st.sidebar.success("File loaded successfully!")
            return df
        else:
            st.sidebar.info("Awaiting file upload...")
            return None
    elif source == "URL":
        url = st.sidebar.text_input("CSV URL")
        if url:
            df = load_url_data(url)
            if df is not None:
                st.sidebar.success("Data loaded from URL!")
            return df
        else:
            st.sidebar.info("Awaiting URL input...")
            return None
    else:
        df = load_kddcup_data()
        if df is not None:
            st.sidebar.success("Sample data loaded!")
        return df

def refresh_data():
    st.cache_data.clear()
    st.experimental_rerun()

# ==============================
# PREPROCESSING & FEATURE SELECTION MODULE
# ==============================
def preprocess_data(df, features):
    """Convert features to numeric, fill missing values (median or 0), and scale."""
    X = df[features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        med = X[col].median()
        if pd.isna(med):
            med = 0
        X[col].fillna(med, inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def apply_preprocessing(df, features, scaler):
    """Apply the same imputation and scaling using an existing scaler."""
    X = df[features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        med = X[col].median()
        if pd.isna(med):
            med = 0
        X[col].fillna(med, inplace=True)
    return scaler.transform(X)

def auto_select_features(df, features, corr_threshold=0.9):
    """Automatically drop features with correlation above threshold."""
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    selected = [feat for feat in features if feat not in to_drop]
    return selected

# ==============================
# DATA SYNOPSIS MODULE
# ==============================
def show_dataset_overview(df):
    n_rows, n_cols = df.shape
    total = n_rows * n_cols
    missing = df.isnull().sum().sum()
    pct = (missing / total) * 100
    cols = st.columns(5)
    cols[0].metric("Rows", n_rows)
    cols[1].metric("Columns", n_cols)
    cols[2].metric("Cells", total)
    cols[3].metric("Missing %", f"{pct:.1f}%")
    cols[4].metric("Avg Unique/Col", f"{int(df.nunique().mean())}")

def data_synopsis(df):
    st.header("Data Synopsis")
    st.subheader("Dataset Overview")
    show_dataset_overview(df)
    
    tabs = st.tabs(["Basic", "Stats", "Missing", "Distributions", "Outliers", "Clustering", "Categorical"])
    with tabs[0]:
        st.markdown("#### Data Types")
        dtypes = df.dtypes.value_counts().reset_index()
        dtypes.columns = ['Data Type', 'Count']
        st.dataframe(dtypes)
    with tabs[1]:
        st.markdown("#### Descriptive Statistics (Numeric)")
        nums = df.select_dtypes(include=np.number).columns.tolist()
        if nums:
            desc = df[nums].describe().T
            desc['median'] = df[nums].median()
            desc['skew'] = df[nums].skew()
            desc['kurtosis'] = df[nums].kurt()
            st.dataframe(desc.style.background_gradient(cmap='viridis'))
        else:
            st.info("No numeric columns.")
    with tabs[2]:
        st.markdown("#### Missing Values")
        n = df.shape[0]
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ['Feature', 'Missing Count']
        missing_df['Missing %'] = (missing_df['Missing Count'] / n * 100).round(2)
        st.dataframe(missing_df.style.background_gradient(cmap='Reds'))
        st.markdown("**Missing Value Heatmap**")
        fig, ax = plt.subplots(figsize=(8, 4))
        msno.heatmap(df, ax=ax)
        st.pyplot(fig)
    with tabs[3]:
        st.markdown("#### Numeric Distributions")
        nums = df.select_dtypes(include=np.number).columns.tolist()
        if nums:
            for col in nums:
                with st.expander(f"Distribution of {col}"):
                    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
                    sns.histplot(df[col].dropna(), kde=True, ax=axs[0], color="#1f77b4")
                    axs[0].set_title("Histogram + KDE")
                    sns.boxplot(x=df[col].dropna(), ax=axs[1], color="#ff7f0e")
                    axs[1].set_title("Box Plot")
                    st.pyplot(fig)
        else:
            st.info("No numeric features.")
    with tabs[4]:
        st.markdown("#### Outlier Analysis (IQR Method)")
        nums = df.select_dtypes(include=np.number).columns.tolist()
        if nums:
            summary = {}
            for col in nums:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower) | (df[col] > upper)][col]
                summary[col] = {"Lower": lower, "Upper": upper, "Outliers": len(outliers)}
            st.dataframe(pd.DataFrame(summary).T.style.background_gradient(cmap='OrRd'))
        else:
            st.info("No numeric columns for outlier detection.")
    with tabs[5]:
        st.markdown("#### Feature Clustering (K-Means)")
        nums = df.select_dtypes(include=np.number).columns.tolist()
        if nums and len(nums) >= 2:
            scaler_temp = StandardScaler()
            scaled = scaler_temp.fit_transform(df[nums].fillna(0))
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(scaled.T)
            cluster_df = pd.DataFrame({"Feature": nums, "Cluster": clusters})
            st.dataframe(cluster_df.sort_values("Cluster"))
            st.markdown("**Clustered Correlation Heatmap**")
            order = cluster_df.sort_values("Cluster")["Feature"].tolist()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df[order].corr(), cmap='coolwarm', annot=True, fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough numeric features for clustering.")
    with tabs[6]:
        st.markdown("#### Categorical Overview")
        cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cats:
            for col in cats:
                with st.expander(col):
                    st.dataframe(df[col].value_counts().to_frame())
                    if df[col].nunique() > 10:
                        text = " ".join(df[col].dropna().astype(str).tolist())
                        wc = WordCloud(width=600, height=300, background_color="white").generate(text)
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.imshow(wc, interpolation="bilinear")
                        ax.axis("off")
                        st.pyplot(fig)
        else:
            st.info("No categorical columns.")

# ==============================
# MODEL TRAINING MODULE
# ==============================
def train_isolation_forest(df, features, contamination, n_estimators, random_state):
    X_scaled, scaler = preprocess_data(df, features)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples='auto',
        bootstrap=True,
        random_state=random_state
    )
    model.fit(X_scaled)
    preds = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)
    return model, preds, scores, scaler

def train_lof(df, features, n_neighbors, contamination):
    X_scaled, scaler = preprocess_data(df, features)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    preds = lof.fit_predict(X_scaled)
    scores = lof.negative_outlier_factor_
    return lof, preds, scores, scaler

def train_ocsvm(df, features, kernel, gamma, contamination, random_state):
    X_scaled, scaler = preprocess_data(df, features)
    ocsvm = OneClassSVM(kernel=kernel, gamma=gamma, nu=contamination)
    ocsvm.fit(X_scaled)
    preds = ocsvm.predict(X_scaled)
    scores = ocsvm.decision_function(X_scaled)
    return ocsvm, preds, scores, scaler

def train_autoencoder(df, features, contamination, epochs=50, batch_size=32, lr=1e-3):
    # If ground truth is available, train only on normal samples.
    if 'true_label' in df.columns:
        train_df = df[df['true_label'].str.lower() == 'normal.']
    else:
        train_df = df.copy()
    
    X_train, scaler = preprocess_data(train_df, features)
    X_all, _ = preprocess_data(df, features)
    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super(Autoencoder, self).__init__()
            hidden1 = input_dim // 2
            hidden2 = input_dim // 4
            bottleneck = max(1, input_dim // 8)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, bottleneck),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, input_dim)
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))
    
    model = Autoencoder(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    dataloader = torch.utils.data.DataLoader(X_tensor, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for inputs in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(X_tensor)
        st.write(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}")
    
    model.eval()
    with torch.no_grad():
        X_tensor_all = torch.tensor(X_all, dtype=torch.float32).to(device)
        X_recon = model(X_tensor_all).cpu().numpy()
    errors = np.mean((X_all - X_recon) ** 2, axis=1)
    threshold = np.percentile(errors, 100 * (1 - contamination))
    preds = np.where(errors > threshold, -1, 1)
    return model, preds, errors, scaler

def train_supervised_rf(df, features, random_state):
    X_scaled, scaler = preprocess_data(df, features)
    y = df['true_label'].apply(lambda x: 1 if x.lower() == 'normal.' else 0).values
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight='balanced')
    clf.fit(X_scaled, y)
    preds = clf.predict(X_scaled)
    mapped_preds = np.where(preds == 1, 1, -1)
    return clf, mapped_preds, scaler

def train_xgboost(df, features, random_state):
    X_scaled, scaler = preprocess_data(df, features)
    y = df['true_label'].apply(lambda x: 1 if x.lower() == 'normal.' else 0).values
    model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                              random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_scaled, y)
    preds = model.predict(X_scaled)
    mapped_preds = np.where(preds == 1, 1, -1)
    return model, mapped_preds, scaler

# ==============================
# EVALUATION MODULE
# ==============================
def evaluate_model(df, preds):
    if 'true_label' not in df.columns:
        st.info("No ground truth available.")
        return
    true_binary = df['true_label'].apply(lambda x: 1 if x.lower() == 'normal.' else -1)
    precision = precision_score(true_binary, preds, pos_label=-1, zero_division=0)
    recall = recall_score(true_binary, preds, pos_label=-1, zero_division=0)
    f1 = f1_score(true_binary, preds, pos_label=-1, zero_division=0)
    cols = st.columns(3)
    cols[0].metric("Precision", f"{precision:.3f}")
    cols[1].metric("Recall", f"{recall:.3f}")
    cols[2].metric("F1 Score", f"{f1:.3f}")
    cm = confusion_matrix(true_binary, preds, labels=[-1, 1])
    cm_df = pd.DataFrame(cm, index=["Actual Anomaly", "Actual Normal"],
                         columns=["Predicted Anomaly", "Predicted Normal"])
    st.markdown("#### Confusion Matrix")
    st.dataframe(cm_df)
    return precision, recall, f1

def compute_feature_importance(df, features, scores):
    st.markdown("#### Feature Importance (Approx.)")
    imp = {feat: abs(np.corrcoef(df[feat].fillna(0), scores)[0, 1]) for feat in features}
    imp_df = pd.DataFrame(imp.items(), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=False)
    st.dataframe(imp_df)
    return imp_df

# ==============================
# VISUALIZATION MODULE
# ==============================
def pca_visualization(df, features, scaler, random_state):
    X_scaled = apply_preprocessing(df, features, scaler)
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Anomaly Label"] = df["Anomaly Label"]
    chart = alt.Chart(pca_df).mark_circle(size=60).encode(
        x=alt.X("PC1", title="PC1"),
        y=alt.Y("PC2", title="PC2"),
        color=alt.Color("Anomaly Label", scale=alt.Scale(domain=["Normal", "Anomaly"],
                                                         range=["#1f77b4", "#d62728"])),
        tooltip=["PC1", "PC2", "Anomaly Label"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

def scatter_visualization(df, features):
    if len(features) < 2:
        st.info("Select at least two features.")
        return
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis", options=features, key="scatter_x")
    with col2:
        y_axis = st.selectbox("Y-axis", options=features, index=1 if len(features) > 1 else 0, key="scatter_y")
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X(x_axis),
        y=alt.Y(y_axis),
        color=alt.Color("Anomaly Label", scale=alt.Scale(domain=["Normal", "Anomaly"],
                                                         range=["#1f77b4", "#d62728"])),
        tooltip=features + ["Anomaly Label", "Anomaly Score"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

def scatter_matrix_visualization(df, features):
    if len(features) < 2:
        st.info("Need at least two features.")
        return
    fig = px.scatter_matrix(df, dimensions=features, color="Anomaly Label",
                            title="Scatter Matrix")
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# MAIN APP LAYOUT
# ==============================
st.title("Network Anomaly Detector Dashboard")
st.markdown("Use the sidebar to choose your data source, model, and adjust settings independently.")

# Data Loading
df = get_data()
if st.sidebar.button("Refresh Data"):
    refresh_data()
if df is None:
    st.stop()

st.subheader("Data Preview")
st.dataframe(df.head(), height=250)

# Sidebar: Model & Feature Options
model_option = st.sidebar.selectbox("Anomaly Model", 
    options=["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "Autoencoder", "Supervised (Random Forest)", "Supervised (XGBoost)"])
auto_fs = st.sidebar.checkbox("Auto Feature Selection", value=True)
corr_thresh = st.sidebar.slider("Correlation Threshold", 0.5, 1.0, 0.9, step=0.05)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if auto_fs:
    selected_features = auto_select_features(df, numeric_cols, corr_thresh)
    st.sidebar.write(f"Auto‑selected features ({len(selected_features)}):", selected_features)
else:
    selected_features = st.sidebar.multiselect("Select Features for Model", options=numeric_cols,
                                                 default=numeric_cols[:min(3, len(numeric_cols))])
if not selected_features:
    st.error("Please select at least one numeric feature.")
    st.stop()

# Additional model parameters
if model_option == "Isolation Forest":
    cont = st.sidebar.slider("Contamination", 0.01, 0.5, 0.1, step=0.01)
    n_trees = st.sidebar.number_input("Number of Trees", 50, 500, 100, step=10)
elif model_option == "Local Outlier Factor":
    cont = st.sidebar.slider("Contamination", 0.01, 0.5, 0.1, step=0.01)
    n_neighbors = st.sidebar.number_input("Number of Neighbors", 10, 100, 20, step=1)
elif model_option == "One-Class SVM":
    cont = st.sidebar.slider("Nu (Contamination)", 0.01, 0.5, 0.1, step=0.01)
    kernel = st.sidebar.selectbox("Kernel", options=["rbf", "linear"])
    gamma = st.sidebar.selectbox("Gamma", options=["scale", "auto"])
elif model_option == "Autoencoder":
    cont = st.sidebar.slider("Contamination (for threshold)", 0.01, 0.5, 0.1, step=0.01)
    epochs = st.sidebar.number_input("Epochs", 10, 200, 50, step=10)
    batch_size = st.sidebar.number_input("Batch Size", 8, 128, 32, step=8)
    lr = st.sidebar.number_input("Learning Rate", 1e-4, 1e-2, 1e-3, step=1e-4, format="%.4f")
elif model_option in ["Supervised (Random Forest)", "Supervised (XGBoost)"]:
    # No extra hyperparameters needed here.
    pass

rand_state = st.sidebar.number_input("Random State", 0, 1000, 42)

# Main Tabs
tabs = st.tabs(["Data Synopsis", "Model & Evaluation", "Visualizations", "Download", "Extra Report"])

with tabs[0]:
    data_synopsis(df)

with tabs[1]:
    st.header("Model Training & Evaluation")
    if model_option == "Isolation Forest":
        with st.spinner("Training Isolation Forest..."):
            model, preds, scores, scaler = train_isolation_forest(df, selected_features, cont, n_trees, rand_state)
        df["Anomaly"] = preds
        df["Anomaly Label"] = df["Anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")
        df["Anomaly Score"] = scores
        st.success("Isolation Forest trained!")
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Anomalies", (df["Anomaly"] == -1).sum())
        colB.metric("Normals", (df["Anomaly"] == 1).sum())
        colC.metric("Contamination", f"{cont:.2f}")
        colD.metric("Trees", n_trees)
    elif model_option == "Local Outlier Factor":
        with st.spinner("Training Local Outlier Factor..."):
            model, preds, scores, scaler = train_lof(df, selected_features, n_neighbors, cont)
        df["Anomaly"] = preds
        df["Anomaly Label"] = df["Anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")
        df["Anomaly Score"] = scores
        st.success("Local Outlier Factor trained!")
        colA, colB, colC = st.columns(3)
        colA.metric("Anomalies", (df["Anomaly"] == -1).sum())
        colB.metric("Normals", (df["Anomaly"] == 1).sum())
        colC.metric("Contamination", f"{cont:.2f}")
    elif model_option == "One-Class SVM":
        with st.spinner("Training One-Class SVM..."):
            model, preds, scores, scaler = train_ocsvm(df, selected_features, kernel, gamma, cont, rand_state)
        df["Anomaly"] = preds
        df["Anomaly Label"] = df["Anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")
        df["Anomaly Score"] = scores
        st.success("One-Class SVM trained!")
        colA, colB, colC = st.columns(3)
        colA.metric("Anomalies", (df["Anomaly"] == -1).sum())
        colB.metric("Normals", (df["Anomaly"] == 1).sum())
        colC.metric("Nu", f"{cont:.2f}")
    elif model_option == "Autoencoder":
        with st.spinner("Training Autoencoder (PyTorch)..."):
            model, preds, errors, scaler = train_autoencoder(df, selected_features, cont, epochs, batch_size, lr)
        df["Anomaly"] = preds
        df["Anomaly Label"] = df["Anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")
        df["Anomaly Score"] = errors
        st.success("Autoencoder trained!")
        col1, col2 = st.columns(2)
        col1.metric("Anomalies", (df["Anomaly"] == -1).sum())
        col2.metric("Normals", (df["Anomaly"] == 1).sum())
    elif model_option == "Supervised (Random Forest)":
        if 'true_label' not in df.columns:
            st.error("Supervised model requires a 'true_label' column.")
            st.stop()
        with st.spinner("Training Supervised Random Forest..."):
            model, preds, scaler = train_supervised_rf(df, selected_features, rand_state)
        df["Anomaly"] = preds
        df["Anomaly Label"] = df["Anomaly"].apply(lambda x: "Normal" if x == 1 else "Anomaly")
        st.success("Supervised Random Forest trained!")
        cols = st.columns(2)
        cols[0].metric("Predicted Normals", (df["Anomaly"] == 1).sum())
        cols[1].metric("Predicted Anomalies", (df["Anomaly"] == -1).sum())
    elif model_option == "Supervised (XGBoost)":
        if 'true_label' not in df.columns:
            st.error("Supervised model requires a 'true_label' column.")
            st.stop()
        with st.spinner("Training Supervised XGBoost..."):
            model, preds, scaler = train_xgboost(df, selected_features, rand_state)
        df["Anomaly"] = preds
        df["Anomaly Label"] = df["Anomaly"].apply(lambda x: "Normal" if x == 1 else "Anomaly")
        st.success("Supervised XGBoost trained!")
        cols = st.columns(2)
        cols[0].metric("Predicted Normals", (df["Anomaly"] == 1).sum())
        cols[1].metric("Predicted Anomalies", (df["Anomaly"] == -1).sum())
    
    st.markdown("---")
    st.subheader("Evaluation Metrics")
    evaluate_model(df, preds)
    st.markdown("---")
    if model_option in ["Isolation Forest", "Local Outlier Factor", "Autoencoder"]:
        compute_feature_importance(df, selected_features, df["Anomaly Score"])
    st.markdown("---")
    st.subheader("PCA Visualization")
    pca_visualization(df, selected_features, scaler, rand_state)

with tabs[2]:
    st.header("Additional Visualizations")
    st.markdown("#### Interactive Scatter Plot")
    scatter_visualization(df, selected_features)
    st.markdown("---")
    st.markdown("#### Scatter Matrix")
    scatter_matrix_visualization(df, selected_features)
    st.markdown("---")
    st.markdown("#### Anomaly Score Density")
    fig, ax = plt.subplots(figsize=(8, 4))
    if "Anomaly Score" in df.columns:
        sns.kdeplot(df["Anomaly Score"], shade=True, color="#d62728", ax=ax)
        ax.set_title("Anomaly Score Density")
        st.pyplot(fig)
    else:
        st.info("Anomaly Score not available.")
    st.markdown("---")
    st.markdown("#### Correlation Matrix")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        st.dataframe(corr.style.background_gradient(cmap='coolwarm'))
    else:
        st.info("Not enough numeric features.")

with tabs[3]:
    st.header("Download Results")
    @st.cache_data(show_spinner=False)
    def convert_df_to_csv(dataframe):
        return dataframe.to_csv(index=False).encode('utf-8')
    csv_data = convert_df_to_csv(df)
    st.download_button(label="Download CSV", data=csv_data,
                       file_name='anomaly_detection_results.csv', mime='text/csv')

with tabs[4]:
    st.header("Extra HTML Report")
    st.markdown("Generate a full HTML report using pandas-profiling (requires installation).")
    try:
        from pandas_profiling import ProfileReport
        if st.button("Generate HTML Report"):
            profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
            html = profile.to_html()
            st.download_button("Download Report", html, file_name="data_profiling_report.html", mime='text/html')
    except ImportError:
        st.warning("Please install pandas-profiling (`pip install pandas-profiling`) to generate reports.")