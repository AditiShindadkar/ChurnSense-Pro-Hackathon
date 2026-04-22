import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="ChurnSense Pro", layout="wide")

st.title("💰 ChurnSense Pro – AI Retention Intelligence System")

st.markdown("""
### 🚀 AI-Powered Customer Retention Platform
Turn raw customer data into **profit-driven decisions** using ML + optimization.

- 🎯 Predict churn risk  
- 💰 Maximize retention ROI  
- 🧩 Segment high-value customers  
""")

st.markdown("🔗 GitHub Repo: https://github.com/AditiShindadkar/ChurnSense-Pro-Hackathon")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Business Settings")

uploaded_file = st.sidebar.file_uploader("Upload Dataset", type="csv")

CLV = st.sidebar.number_input("Customer Lifetime Value", value=500)
RETENTION_COST = st.sidebar.number_input("Retention Cost", value=50)
SUCCESS_RATE = st.sidebar.slider("Success Rate", 0.1, 1.0, 0.5)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data(file):
    if file:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv("customer_churn_dataset-training-master.csv")

    if "CustomerID" in df.columns:
        df = df.drop("CustomerID", axis=1)

    df = df.dropna()

    cat_cols = ['Gender', 'Subscription Type', 'Contract Length']
    existing = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42), X

# -----------------------------
# TRAIN MODEL
# -----------------------------
@st.cache_resource
def train_model(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    model = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100)),
            ('xgb', XGBClassifier(eval_metric='logloss'))
        ],
        voting='soft'
    )

    model.fit(X_scaled, y_res)
    return model, scaler

# -----------------------------
# MAIN EXECUTION
# -----------------------------
(X_train, X_test, y_train, y_test), X_full = load_data(uploaded_file)

model, scaler = train_model(X_train, y_train)

X_test_scaled = scaler.transform(X_test)
y_probs = model.predict_proba(X_test_scaled)[:, 1]

# -----------------------------
# PROFIT FUNCTION
# -----------------------------
def calculate_profit(threshold):
    preds = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    revenue = tp * CLV * SUCCESS_RATE
    cost = (tp + fp) * RETENTION_COST

    return revenue - cost

thresholds = np.arange(0, 1, 0.05)
profits = [calculate_profit(t) for t in thresholds]

opt_th = thresholds[np.argmax(profits)]
max_profit = max(profits)

# -----------------------------
# METRICS
# -----------------------------
c1, c2, c3 = st.columns(3)

c1.metric("💰 Max Profit", f"${max_profit:,.0f}", delta="Optimized ROI")
c2.metric("🎯 Optimal Threshold", f"{opt_th:.2f}", delta="Decision Boundary")
c3.metric("👥 Target Customers", int((y_probs >= opt_th).sum()), delta="Actionable Users")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Profit", "🧩 Segmentation", "📋 Actions", "🎯 Predict"]
)

# -----------------------------
# TAB 1: PROFIT
# -----------------------------
with tab1:
    st.subheader("Profit Optimization Curve")

    fig, ax = plt.subplots()
    ax.plot(thresholds, profits, color="green")
    ax.axvline(opt_th, color="red", linestyle="--")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Profit")
    st.pyplot(fig)

# -----------------------------
# TAB 2: SEGMENTATION
# -----------------------------
with tab2:
    st.subheader("Customer Segmentation")

    idx = np.where(y_probs >= opt_th)[0]

    if len(idx) > 0:
        df_seg = X_test.iloc[idx].copy()

        try:
            cols = ['Tenure', 'Monthly Bill']
            kmeans = KMeans(n_clusters=3, random_state=42)
            df_seg["Cluster"] = kmeans.fit_predict(df_seg[cols])

            fig2, ax2 = plt.subplots()
            sns.scatterplot(
                data=df_seg,
                x=cols[0],
                y=cols[1],
                hue="Cluster",
                palette="viridis",
                ax=ax2
            )
            st.pyplot(fig2)

            st.markdown("""
### 🧠 Strategy Guide
- 🔴 High Value + High Risk → VIP Retention (Call + Offer)
- 🟡 Medium Risk → Discounts
- 🟢 Low Risk → No action
""")

        except Exception as e:
            st.warning(f"Segmentation failed: {e}")

# -----------------------------
# TAB 3: ACTION LIST
# -----------------------------
with tab3:
    st.subheader("Retention Action List")

    df_res = X_test.copy()
    df_res["Risk"] = y_probs
    df_res["Action"] = df_res["Risk"].apply(
        lambda x: "SEND OFFER" if x >= opt_th else "IGNORE"
    )

    high_risk = df_res[df_res["Action"] == "SEND OFFER"]

    st.dataframe(high_risk.head(10))

    csv = high_risk.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Target List",
        csv,
        "targets.csv",
        "text/csv"
    )

# -----------------------------
# TAB 4: INDIVIDUAL PREDICTION
# -----------------------------
with tab4:
    st.subheader("Predict Individual Customer")

    sample = X_train.iloc[0:1]

    inputs = {}
    for col in sample.columns:
        inputs[col] = st.number_input(col, value=float(sample[col].values[0]))

    if st.button("Predict"):
        df_input = pd.DataFrame([inputs])
        df_input = df_input.reindex(columns=X_train.columns, fill_value=0)

        scaled = scaler.transform(df_input)
        prob = model.predict_proba(scaled)[:, 1][0]

        st.success(f"Churn Probability: {prob:.2f}")

        if prob > opt_th:
            st.error("⚠️ High Risk Customer → Immediate Action Required")
        else:
            st.success("✅ Low Risk Customer → No Action Needed")

        expected_loss = prob * CLV
        st.info(f"💸 Expected Revenue Loss: ${expected_loss:.2f}")

# -----------------------------
# FINAL INSIGHT
# -----------------------------
st.markdown("---")
st.markdown("### 💡 Insight")
st.success(
    "Instead of targeting all customers, this system ensures every ₹1 spent on retention maximizes ROI."
)
