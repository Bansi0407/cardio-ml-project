# Name : Bansi Ajagia
# Subject : Machine Learning
# Project : Cardiovascular Disease Prediction System

---

# ❤️ Cardiovascular Disease Prediction System

A complete end-to-end Machine Learning project built with **Python**, **Jupyter Notebook**, and **Streamlit**.

## 📌 Project Overview

This project predicts whether a patient is at risk of cardiovascular disease using clinical and lifestyle data.
It covers the full ML pipeline: data exploration → preprocessing → model training → evaluation → deployment.

## 📦 Dataset

- **Source**: [Cardiovascular Disease Dataset — Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- **Records**: ~70,000 patients
- **Features**: 11 (age, gender, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol, activity)
- **Target**: `cardio` (0 = No Disease, 1 = Disease)

## 🗂️ Project Structure

```
cardio_ml_project/
├── app.py              # Streamlit frontend
├── model.py            # ML model training & prediction
├── utils.py            # Data loading & preprocessing
├── config.py           # Project configuration
├── requirements.txt    # Python dependencies
├── README.md
├── data/
│   └── cardio.csv      # ← Place dataset here
├── models/
│   └── model.pkl       # Saved trained model
├── outputs/
│   ├── metrics.txt
│   └── confusion_matrix.png
└── tasks/              # Jupyter Notebooks (10 tasks)
    ├── task1_problem.ipynb
    ├── task2_preprocessing.ipynb
    ├── task3_model.ipynb
    ├── task4_evaluation.ipynb
    ├── task5_advanced.ipynb
    ├── task6_visualization.ipynb
    ├── task7_streamlit_setup.ipynb
    ├── task8_frontend.ipynb
    ├── task9_backend.ipynb
    └── task10_deployment.ipynb
```

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place the dataset
Download `cardio_train.csv` from Kaggle, rename it to `cardio.csv`, and place it in the `data/` folder.

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### 4. Open in browser
Visit: **http://localhost:8501**

## 🤖 ML Models Used

| Model | Type |
|---|---|
| Logistic Regression | Library (sklearn) |
| Random Forest | Library (sklearn) |
| Rule-Based Classifier | Manual (no library) |

## 📊 Streamlit App Pages

- 🏠 **Home** — Project overview
- 📊 **Data Analysis** — EDA, plots, heatmap
- 🤖 **Train Model** — Train & compare models
- 🔮 **Prediction** — Enter patient data → get risk prediction

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Visit https://share.streamlit.io
3. Connect your GitHub repo
4. Set main file to `app.py` and deploy

---
*B.Tech CSE Semester 6 — Machine Learning Project*
