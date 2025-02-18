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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import fetch_kddcup99

# Unsupervised Models
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Supervised Models
import xgboost as xgb

# PyTorch for Autoencoder
import torch
import torch.nn as nn
import torch.optim as optim

# ========= GLOBAL CONFIGURATION & STYLING =========
sns.set_theme(style="whitegrid")
st.set_page_config(page_title="Network Anomaly Detector Dashboard", layout="wide", initial_sidebar_state="expanded")

# ========= DATA LOADING =========
@st.cache_data(show_spinner=True)
def load_csv(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error: {e}")
        return None

@st.cache_data(show_spinner=True)
def load_url(url):
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return pd.read_csv(StringIO(resp.text))
    except Exception as e:
        st.error(f"Error: {e}")
        return None

@st.cache_data(show_spinner=True)
def load_sample():
    st.info("Loading KDD Cup 99 sample data...")
    try:
        data = fetch_kddcup99(percent10=True, subset='SA', random_state=42)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['true_label'] = [label.decode('utf-8') for label in data.target]
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def get_data():
    src = st.sidebar.radio("Data Source", ("Upload File", "URL", "Sample Data"))
    if src == "Upload File":
        file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        return load_csv(file) if file else None
    elif src == "URL":
        url = st.sidebar.text_input("CSV URL")
        return load_url(url) if url else None
    else:
        return load_sample()

def refresh_data():
    st.cache_data.clear()
    st.experimental_rerun()

# ========= PREPROCESSING & FEATURE SELECTION =========
def preprocess(df, features):
    """Convert features to numeric, fill missing (median or 0), and scale."""
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

def apply_scaler(df, features, scaler):
    X = df[features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        med = X[col].median()
        if pd.isna(med):
            med = 0
        X[col].fillna(med, inplace=True)
    return scaler.transform(X)

def auto_select(df, features, threshold=0.9):
    corr = df[features].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return [f for f in features if f not in drop]

# ========= DATA ANALYSIS SYNOPSIS =========
def data_overview(df):
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
    st.header("Data Analysis Synopsis")
    st.subheader("Overview")
    data_overview(df)
    tabs = st.tabs(["Data Types", "Descriptive Stats", "Missing Values", "Distributions", "Outliers", "Correlation"])
    
    with tabs[0]:
        st.write("**Data Types**")
        st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Feature", 0:"Type"}))
        
    with tabs[1]:
        st.write("**Descriptive Statistics**")
        nums = df.select_dtypes(include=np.number).columns.tolist()
        if nums:
            st.dataframe(df[nums].describe().T)
        else:
            st.info("No numeric columns available.")
    
    with tabs[2]:
        st.write("**Missing Values**")
        miss = df.isnull().sum().reset_index().rename(columns={"index": "Feature", 0:"Missing"})
        st.dataframe(miss)
        fig, ax = plt.subplots(figsize=(8,4))
        msno.heatmap(df, ax=ax)
        st.pyplot(fig)
    
    with tabs[3]:
        st.write("**Numeric Distributions**")
        nums = df.select_dtypes(include=np.number).columns.tolist()
        for col in nums:
            with st.expander(f"Distribution of {col}"):
                fig, ax = plt.subplots(1,2, figsize=(12,4))
                sns.histplot(df[col].dropna(), kde=True, ax=ax[0], color="#1f77b4")
                ax[0].set_title(f"{col} Histogram")
                sns.boxplot(x=df[col].dropna(), ax=ax[1], color="#ff7f0e")
                ax[1].set_title(f"{col} Boxplot")
                st.pyplot(fig)
    
    with tabs[4]:
        st.write("**Outlier Analysis (IQR)**")
        nums = df.select_dtypes(include=np.number).columns.tolist()
        out_summary = {}
        for col in nums:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            out_summary[col] = {"Lower": lower, "Upper": upper, "Outliers": ((df[col] < lower) | (df[col] > upper)).sum()}
        st.dataframe(pd.DataFrame(out_summary).T)
    
    with tabs[5]:
        st.write("**Correlation Matrix**")
        nums = df.select_dtypes(include=np.number).columns.tolist()
        if nums:
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(df[nums].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough numeric features.")

# ========= MODELING FUNCTIONS =========

def model_iforest(df, features, contamination, n_estimators, random_state):
    X_scaled, scaler = preprocess(df, features)
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                             bootstrap=True, random_state=random_state)
    model.fit(X_scaled)
    preds = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)
    return model, preds, scores, scaler

def model_lof(df, features, n_neighbors, contamination):
    X_scaled, scaler = preprocess(df, features)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    preds = lof.fit_predict(X_scaled)
    scores = lof.negative_outlier_factor_
    return lof, preds, scores, scaler

def model_ocsvm(df, features, kernel, gamma, contamination, random_state):
    X_scaled, scaler = preprocess(df, features)
    ocsvm = OneClassSVM(kernel=kernel, gamma=gamma, nu=contamination)
    ocsvm.fit(X_scaled)
    preds = ocsvm.predict(X_scaled)
    scores = ocsvm.decision_function(X_scaled)
    return ocsvm, preds, scores, scaler

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, input_dim//4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim//4, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def model_autoencoder(df, features, contamination, epochs, batch_size, lr):
    train_df = df[df['true_label'].str.lower()=='normal.'] if 'true_label' in df.columns else df.copy()
    X_train, scaler = preprocess(train_df, features)
    X_all, _ = preprocess(df, features)
    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        st.write(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss/len(train_tensor):.6f}")
    model.eval()
    with torch.no_grad():
        all_tensor = torch.tensor(X_all, dtype=torch.float32).to(device)
        recon = model(all_tensor).cpu().numpy()
    errors = np.mean((X_all - recon)**2, axis=1)
    threshold = np.percentile(errors, 100 * (1 - contamination))
    preds = np.where(errors > threshold, -1, 1)
    return model, preds, errors, scaler

def model_rf(df, features, random_state):
    from sklearn.ensemble import RandomForestClassifier
    X_scaled, scaler = preprocess(df, features)
    y = df['true_label'].apply(lambda x: 1 if x.lower()=='normal.' else 0).values
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight='balanced')
    clf.fit(X_scaled, y)
    preds = clf.predict(X_scaled)
    mapped = np.where(preds==1, 1, -1)
    return clf, mapped, scaler

def model_xgb(df, features, random_state):
    X_scaled, scaler = preprocess(df, features)
    y = df['true_label'].apply(lambda x: 1 if x.lower()=='normal.' else 0).values
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                            random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_scaled, y)
    preds = clf.predict(X_scaled)
    mapped = np.where(preds==1, 1, -1)
    return clf, mapped, scaler

# ========= EVALUATION =========
def evaluate_model(df, preds):
    if 'true_label' not in df.columns:
        st.info("No ground truth available.")
        return
    y_true = df['true_label'].apply(lambda x: 1 if x.lower()=='normal.' else -1)
    prec = precision_score(y_true, preds, pos_label=-1, zero_division=0)
    rec = recall_score(y_true, preds, pos_label=-1, zero_division=0)
    f1 = f1_score(y_true, preds, pos_label=-1, zero_division=0)
    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{prec:.3f}")
    col2.metric("Recall", f"{rec:.3f}")
    col3.metric("F1 Score", f"{f1:.3f}")
    cm = confusion_matrix(y_true, preds, labels=[-1, 1])
    st.dataframe(pd.DataFrame(cm, index=["Actual Anomaly", "Actual Normal"],
                              columns=["Predicted Anomaly", "Predicted Normal"]))
    return prec, rec, f1

def feature_importance(df, features, scores):
    imp = {feat: abs(np.corrcoef(df[feat].fillna(0), scores)[0,1]) for feat in features}
    imp_df = pd.DataFrame(imp.items(), columns=["Feature","Importance"]).sort_values(by="Importance", ascending=False)
    st.dataframe(imp_df)
    return imp_df

# ========= VISUALIZATION =========
def pca_plot(df, features, scaler, random_state=42):
    X_scaled = apply_scaler(df, features, scaler)
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Anomaly Label"] = df["Anomaly Label"]
    chart = alt.Chart(pca_df).mark_circle(size=60).encode(
        x=alt.X("PC1", title="Principal Component 1"),
        y=alt.Y("PC2", title="Principal Component 2"),
        color=alt.Color("Anomaly Label", scale=alt.Scale(domain=["Normal","Anomaly"],
                                                         range=["#1f77b4","#d62728"])),
        tooltip=["PC1", "PC2", "Anomaly Label"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

def plot_scatter(df, features):
    if len(features) < 2:
        st.info("Select at least two features.")
        return
    col1, col2 = st.columns(2)
    x_axis = col1.selectbox("X-axis", options=features, key="x_scatter")
    y_axis = col2.selectbox("Y-axis", options=features, key="y_scatter")
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X(x_axis, title=x_axis),
        y=alt.Y(y_axis, title=y_axis),
        color=alt.Color("Anomaly Label", scale=alt.Scale(domain=["Normal", "Anomaly"],
                                                         range=["#1f77b4","#d62728"])),
        tooltip=features + ["Anomaly Label", "Anomaly Score"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

def plot_scatter_matrix(df, features):
    if len(features) < 2:
        st.info("Select at least two features.")
        return
    fig = px.scatter_matrix(df, dimensions=features, color="Anomaly Label", title="Scatter Matrix")
    st.plotly_chart(fig, use_container_width=True)

# ========= MAIN DASHBOARD =========
st.title("Network Anomaly Detector Dashboard")
st.markdown("An enhanced, modular dashboard that provides detailed data insights and model analysis.")

df = get_data()
if st.sidebar.button("Refresh Data"):
    refresh_data()
if df is None:
    st.stop()

st.subheader("Data Preview")
st.dataframe(df.head(), height=250)

# Sidebar: Feature and Model Options
model_choice = st.sidebar.selectbox("Choose Model", 
    options=["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "Autoencoder", "Supervised (Random Forest)", "Supervised (XGBoost)"])
auto_fs = st.sidebar.checkbox("Auto Feature Selection", value=True)
corr_thresh = st.sidebar.slider("Correlation Threshold", 0.5, 1.0, 0.9, step=0.05)

num_features = df.select_dtypes(include=np.number).columns.tolist()
if auto_fs:
    features = auto_select(df, num_features, corr_thresh)
    st.sidebar.write(f"Auto-selected ({len(features)}):", features)
else:
    features = st.sidebar.multiselect("Select Features", options=num_features, default=num_features[:min(3, len(num_features))])
if not features:
    st.error("Please select at least one feature.")
    st.stop()

# Model hyperparameters
if model_choice == "Isolation Forest":
    cont = st.sidebar.slider("Contamination", 0.01, 0.5, 0.1, step=0.01)
    n_trees = st.sidebar.number_input("Number of Trees", 50, 500, 100, step=10)
elif model_choice == "Local Outlier Factor":
    cont = st.sidebar.slider("Contamination", 0.01, 0.5, 0.1, step=0.01)
    n_neighbors = st.sidebar.number_input("Number of Neighbors", 10, 100, 20, step=1)
elif model_choice == "One-Class SVM":
    cont = st.sidebar.slider("Nu (Contamination)", 0.01, 0.5, 0.1, step=0.01)
    kernel = st.sidebar.selectbox("Kernel", options=["rbf","linear"])
    gamma = st.sidebar.selectbox("Gamma", options=["scale","auto"])
elif model_choice == "Autoencoder":
    cont = st.sidebar.slider("Contamination (for threshold)", 0.01, 0.5, 0.1, step=0.01)
    epochs = st.sidebar.number_input("Epochs", 10, 200, 50, step=10)
    batch_size = st.sidebar.number_input("Batch Size", 8, 128, 32, step=8)
    lr = st.sidebar.number_input("Learning Rate", 1e-4, 1e-2, 1e-3, step=1e-4, format="%.4f")
elif model_choice in ["Supervised (Random Forest)", "Supervised (XGBoost)"]:
    pass

rand_state = st.sidebar.number_input("Random State", 0, 1000, 42)

# ========= MAIN TABS =========
tabs = st.tabs(["Data Synopsis", "Model & Evaluation", "Visualizations", "Download", "Extra Report"])

with tabs[0]:
    data_synopsis(df)

with tabs[1]:
    st.header("Model Synopsis & Evaluation")
    if model_choice == "Isolation Forest":
        with st.spinner("Training Isolation Forest..."):
            model, preds, scores, scaler = model_iforest(df, features, cont, n_trees, rand_state)
        df["Anomaly"] = preds
        df["Anomaly Label"] = df["Anomaly"].apply(lambda x: "Anomaly" if x==-1 else "Normal")
        df["Anomaly Score"] = scores
        st.success("Isolation Forest trained!")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Anomalies", (df["Anomaly"]==-1).sum())
        c2.metric("Normals", (df["Anomaly"]==1).sum())
        c3.metric("Contamination", f"{cont:.2f}")
        c4.metric("Trees", n_trees)
    elif model_choice == "Local Outlier Factor":
        with st.spinner("Training LOF..."):
            model, preds, scores, scaler = model_lof(df, features, n_neighbors, cont)
        df["Anomaly"] = preds
        df["Anomaly Label"] = df["Anomaly"].apply(lambda x: "Anomaly" if x==-1 else "Normal")
        df["Anomaly Score"] = scores
        st.success("LOF trained!")
        c1, c2, c3 = st.columns(3)
        c1.metric("Anomalies", (df["Anomaly"]==-1).sum())
        c2.metric("Normals", (df["Anomaly"]==1).sum())
        c3.metric("Contamination", f"{cont:.2f}")
    elif model_choice == "One-Class SVM":
        with st.spinner("Training One-Class SVM..."):
            model, preds, scores, scaler = model_ocsvm(df, features, kernel, gamma, cont, rand_state)
        df["Anomaly"] = preds
        df["Anomaly Label"] = df["Anomaly"].apply(lambda x: "Anomaly" if x==-1 else "Normal")
        df["Anomaly Score"] = scores
        st.success("One-Class SVM trained!")
        c1, c2, c3 = st.columns(3)
        c1.metric("Anomalies", (df["Anomaly"]==-1).sum())
        c2.metric("Normals", (df["Anomaly"]==1).sum())
        c3.metric("Nu", f"{cont:.2f}")
    elif model_choice == "Autoencoder":
        with st.spinner("Training Autoencoder..."):
            model, preds, errors, scaler = model_autoencoder(df, features, cont, epochs, batch_size, lr)
        df["Anomaly"] = preds
        df["Anomaly Label"] = df["Anomaly"].apply(lambda x: "Anomaly" if x==-1 else "Normal")
        df["Anomaly Score"] = errors
        st.success("Autoencoder trained!")
        colA, colB = st.columns(2)
        colA.metric("Anomalies", (df["Anomaly"]==-1).sum())
        colB.metric("Normals", (df["Anomaly"]==1).sum())
    elif model_choice == "Supervised (Random Forest)":
        if 'true_label' not in df.columns:
            st.error("Ground truth ('true_label') is required for supervised models.")
            st.stop()
        with st.spinner("Training Random Forest..."):
            model, preds, scaler = model_rf(df, features, rand_state)
        df["Anomaly"] = preds
        df["Anomaly Label"] = df["Anomaly"].apply(lambda x: "Normal" if x==1 else "Anomaly")
        st.success("Random Forest trained!")
        col1, col2 = st.columns(2)
        col1.metric("Predicted Normals", (df["Anomaly"]==1).sum())
        col2.metric("Predicted Anomalies", (df["Anomaly"]==-1).sum())
    elif model_choice == "Supervised (XGBoost)":
        if 'true_label' not in df.columns:
            st.error("Ground truth ('true_label') is required for supervised models.")
            st.stop()
        with st.spinner("Training XGBoost..."):
            model, preds, scaler = model_xgb(df, features, rand_state)
        df["Anomaly"] = preds
        df["Anomaly Label"] = df["Anomaly"].apply(lambda x: "Normal" if x==1 else "Anomaly")
        st.success("XGBoost trained!")
        col1, col2 = st.columns(2)
        col1.metric("Predicted Normals", (df["Anomaly"]==1).sum())
        col2.metric("Predicted Anomalies", (df["Anomaly"]==-1).sum())
    
    st.markdown("---")
    st.subheader("Evaluation Metrics")
    evaluate_model(df, preds)
    st.markdown("---")
    if model_choice in ["Isolation Forest", "Local Outlier Factor", "Autoencoder"]:
        st.subheader("Feature Importance")
        feature_importance(df, features, df["Anomaly Score"])
    st.markdown("---")
    st.subheader("PCA Visualization")
    pca_plot(df, features, scaler, rand_state)

with tabs[2]:
    st.header("Visualizations")
    st.markdown("#### Interactive Scatter Plot")
    plot_scatter(df, features)
    st.markdown("---")
    st.markdown("#### Scatter Matrix")
    plot_scatter_matrix(df, features)
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
    if len(num_features) >= 2:
        corr = df[num_features].corr()
        st.dataframe(corr.style.background_gradient(cmap='coolwarm'))
    else:
        st.info("Not enough numeric features.")

with tabs[3]:
    st.header("Download")
    @st.cache_data(show_spinner=False)
    def df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    csv_data = df_to_csv(df)
    st.download_button("Download CSV", csv_data, file_name="anomaly_results.csv", mime="text/csv")

with tabs[4]:
    st.header("Extra Report")
    st.markdown("Generate a full HTML report with pandas-profiling (requires installation).")
    try:
        from pandas_profiling import ProfileReport
        if st.button("Generate Report"):
            profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
            html = profile.to_html()
            st.download_button("Download Report", html, file_name="data_report.html", mime="text/html")
    except ImportError:
        st.warning("Please install pandas-profiling to generate reports.")
