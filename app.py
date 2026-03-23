# Name : Bansi Ajagia
# Subject : Machine Learning
# Project : Cardiovascular Disease Prediction System

# ============================================================
# app.py — Streamlit frontend (run: streamlit run app.py)
# ============================================================

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import (DATA_PATH, MODEL_PATH, CONF_MATRIX_PATH,
                    METRICS_PATH, CHOL_MAP, GLUC_MAP)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Cardio Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: white;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-card h3 { font-size: 2rem; margin: 0; color: #e94560; }
.metric-card p  { margin: 0; color: #a0aec0; font-size: 0.85rem; }

.risk-high {
    background: linear-gradient(135deg, #742020, #c0392b);
    border-radius: 16px; padding: 2rem; color: white; text-align: center;
}
.risk-none {
    background: linear-gradient(135deg, #1a472a, #27ae60);
    border-radius: 16px; padding: 2rem; color: white; text-align: center;
}
.section-header {
    border-left: 4px solid #e94560;
    padding-left: 0.75rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=80)
st.sidebar.markdown("## ❤️ Cardio Predict")
st.sidebar.markdown("*B.Tech CSE — ML Project*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Data Analysis", "🤖 Train Model", "🔮 Prediction"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Student:** Bansi Ajagia  \n**Subject:** Machine Learning")


# ════════════════════════════════════════════════════════════
# 🏠 HOME PAGE
# ════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("<h1 style='text-align:center'>❤️ Cardiovascular Disease<br>Prediction System</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#718096'>B.Tech CSE Semester 6 — Machine Learning Project</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-header"><h3>About the Project</h3></div>', unsafe_allow_html=True)
        st.markdown("""
Cardiovascular disease (CVD) is the **leading cause of death globally**, accounting for ~17.9 million
deaths per year according to WHO. Early detection is critical.

This system uses **Machine Learning** (Logistic Regression + Random Forest) trained on a large clinical
dataset to predict whether a patient is at risk of cardiovascular disease based on lifestyle and
health metrics.

**Key Features:**
- 🔬 Exploratory Data Analysis dashboard  
- 🤖 Multiple ML models trained & compared  
- ✅ Real-time prediction with confidence  
- 📈 Full evaluation metrics  
        """)

    with col2:
        st.markdown('<div class="section-header"><h3>Dataset Info</h3></div>', unsafe_allow_html=True)
        st.markdown("""
| Property | Value |
|---|---|
| Source | Kaggle |
| Records | ~70 000 |
| Features | 11 |
| Target | `cardio` (0/1) |
| Type | Classification |
        """)
        st.info("📥 Dataset: [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.success("✅ Data Preprocessing Complete")
    c2.success("✅ ML Models Implemented")
    c3.success("✅ Streamlit UI Ready")


# ════════════════════════════════════════════════════════════
# 📊 DATA ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "📊 Data Analysis":
    st.markdown('<div class="section-header"><h2>📊 Data Analysis</h2></div>', unsafe_allow_html=True)

    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at `{DATA_PATH}`. Please place `cardio.csv` in the `data/` folder.")
        st.stop()

    df = pd.read_csv(DATA_PATH, sep=";")
    df["age_years"] = (df["age"] / 365).round(0).astype(int)

    # Quick stats
    st.markdown("### Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Features", str(df.shape[1]))
    c3.metric("Disease Cases", f"{df['cardio'].sum():,}")
    c4.metric("Healthy Cases", f"{(df['cardio']==0).sum():,}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📈 Distributions", "🌡️ Correlation"])

    with tab1:
        st.dataframe(df.head(200), use_container_width=True)
        st.caption(f"Showing first 200 of {len(df):,} rows")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            df["cardio"].value_counts().plot.pie(
                labels=["No Disease", "Disease"],
                autopct="%1.1f%%", colors=["#27ae60", "#e74c3c"], ax=ax,
                startangle=90)
            ax.set_title("Target Class Distribution")
            ax.set_ylabel("")
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df["age_years"], bins=30, color="#0f3460", edgecolor="white")
            ax.set_xlabel("Age (years)")
            ax.set_ylabel("Count")
            ax.set_title("Age Distribution")
            st.pyplot(fig)
            plt.close()

        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x="cholesterol", hue="cardio", palette=["#27ae60","#e74c3c"], ax=ax)
            ax.set_xticklabels(["Normal", "Above Normal", "Well Above"])
            ax.set_title("Cholesterol vs Cardio")
            ax.legend(["No Disease","Disease"])
            st.pyplot(fig)
            plt.close()

        with col4:
            fig, ax = plt.subplots(figsize=(6, 4))
            bp_data = [df[df["cardio"]==0]["ap_hi"].clip(60,200),
                       df[df["cardio"]==1]["ap_hi"].clip(60,200)]
            ax.boxplot(bp_data, labels=["No Disease","Disease"], patch_artist=True,
                       boxprops=dict(facecolor="#0f3460", color="white"),
                       medianprops=dict(color="#e94560", linewidth=2))
            ax.set_title("Systolic BP vs Cardio")
            ax.set_ylabel("ap_hi (mmHg)")
            st.pyplot(fig)
            plt.close()

    with tab3:
        num_cols = ["age_years","height","weight","ap_hi","ap_lo","cholesterol","gluc","cardio"]
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, linewidths=0.5, square=True)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)
        plt.close()


# ════════════════════════════════════════════════════════════
# 🤖 TRAIN MODEL
# ════════════════════════════════════════════════════════════
elif page == "🤖 Train Model":
    st.markdown('<div class="section-header"><h2>🤖 Train Model</h2></div>', unsafe_allow_html=True)

    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at `{DATA_PATH}`.")
        st.stop()

    st.markdown("Click the button below to train **Logistic Regression**, **Random Forest**, and the **Rule-Based** classifier.")

    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        with st.spinner("Training models — please wait…"):
            from model import train_and_save
            results, best_name = train_and_save()

        st.success(f"✅ Training complete! Best model: **{best_name}**")
        st.balloons()

        # Show metrics table
        rows = []
        for name, m in results.items():
            rows.append({
                "Model": name,
                "Accuracy":  round(m["accuracy"],  4),
                "Precision": round(m["precision"], 4),
                "Recall":    round(m["recall"],    4),
                "F1 Score":  round(m["f1"],        4),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

        # Bar chart comparison
        fig, ax = plt.subplots(figsize=(8, 4))
        df_plot = pd.DataFrame(rows).set_index("Model")
        df_plot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="black")
        ax.set_ylim(0, 1)
        ax.set_title("Model Comparison")
        ax.set_xticklabels(df_plot.index, rotation=15)
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close()

        # Confusion matrix
        if os.path.exists(CONF_MATRIX_PATH):
            st.image(CONF_MATRIX_PATH, caption=f"Confusion Matrix — {best_name}", width=420)

    elif os.path.exists(METRICS_PATH):
        st.info("Model already trained. Showing saved metrics.")
        with open(METRICS_PATH) as f:
            st.text(f.read())
        if os.path.exists(CONF_MATRIX_PATH):
            st.image(CONF_MATRIX_PATH, width=420)


# ════════════════════════════════════════════════════════════
# 🔮 PREDICTION
# ════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.markdown('<div class="section-header"><h2>🔮 Cardiovascular Risk Prediction</h2></div>',
                unsafe_allow_html=True)

    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ No trained model found. Please go to **Train Model** first.")
        st.stop()

    st.markdown("Fill in the patient details below and press **Predict**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🧍 Personal Info**")
        age       = st.slider("Age (years)", 10, 100, 45)
        gender    = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
        height    = st.slider("Height (cm)", 100, 220, 165)
        weight    = st.slider("Weight (kg)", 30, 200, 70)

    with col2:
        st.markdown("**🩺 Clinical Measures**")
        ap_hi = st.slider("Systolic BP (ap_hi)", 60, 250, 120)
        ap_lo = st.slider("Diastolic BP (ap_lo)", 40, 200, 80)
        cholesterol = st.selectbox("Cholesterol", [1, 2, 3],
                                   format_func=lambda x: CHOL_MAP[x])
        gluc        = st.selectbox("Glucose", [1, 2, 3],
                                   format_func=lambda x: GLUC_MAP[x])

    with col3:
        st.markdown("**🚬 Lifestyle**")
        smoke  = st.selectbox("Smoking",           [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        alco   = st.selectbox("Alcohol Intake",    [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        active = st.selectbox("Physical Activity", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")

    st.markdown("---")

    if st.button("❤️ Predict Cardiovascular Risk", type="primary", use_container_width=True):
        input_data = {
            "age_years": age, "gender": gender,
            "height": height, "weight": weight,
            "ap_hi": ap_hi, "ap_lo": ap_lo,
            "cholesterol": cholesterol, "gluc": gluc,
            "smoke": smoke, "alco": alco, "active": active,
        }

        from model import predict_single
        result = predict_single(input_data)

        st.markdown("---")
        if result == 1:
            st.markdown("""
<div class="risk-high">
  <h2>⚠️ High Risk of Cardiovascular Disease</h2>
  <p>Based on the provided health metrics, this patient shows indicators of cardiovascular risk.<br>
  <strong>Please consult a medical professional immediately.</strong></p>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="risk-none">
  <h2>✅ No Cardiovascular Disease Detected</h2>
  <p>Based on the provided health metrics, this patient does not show strong indicators of cardiovascular disease.<br>
  <strong>Maintain a healthy lifestyle!</strong></p>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.caption("⚠️ This prediction is for educational purposes only and is not a medical diagnosis.")
