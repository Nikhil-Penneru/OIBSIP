# Credit Card Fraud Detection Pipeline

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-blue?style=for-the-badge)](https://imbalanced-learn.org/)

---

## 📌 Project Overview
This repository contains the complete, submission-ready implementation for **OASIS INFOBYTE Data Analytics Internship Task 3: Fraud Detection**.

The project builds a machine learning pipeline to detect fraudulent credit card transactions from a heavily imbalanced transaction dataset (284,807 transactions, 492 fraud cases, ~0.173% fraud prevalence), treating **class imbalance** as the central challenge.

---

## 📦 Dataset Information

This project uses the **Credit Card Fraud Detection** benchmark dataset
provided by the **Machine Learning Group (MLG) at Université Libre de
Bruxelles (ULB) and Worldline**.

- **Dataset:** Credit Card Fraud Detection
- **Total Transactions:** 284,807
- **Legitimate Transactions (Class 0):** 284,315 (99.827%)
- **Fraudulent Transactions (Class 1):** 492 (0.173%)
- **Imbalance Ratio:** ~578 : 1
- **Features:** 30 numerical input features (`Time`, `V1` to `V28`, `Amount`)
  and binary target `Class`.

### Dataset Source

The dataset was obtained from the following Kaggle benchmark:

**Credit Card Fraud Detection — MLG-ULB**

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The original `creditcard.csv` file is approximately **143 MB** and is not
included in this repository because it exceeds GitHub's single-file size
limit.

To run the notebook locally, download the dataset from the source above
and place `creditcard.csv` in the project directory.

---

## 🏗️ Pipeline Architecture & Methodology

```
┌─────────────────────────┐
│ 1. Raw Dataset (284.8k) │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  2. Data Quality Audit  │ ──> Missing: 0 | Duplicates: 1,081 (0.38%)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 3. Stratified Split 80% │ ──> Train: 227,845 (394 frauds) | Test: 56,962 (98 frauds)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 4. Leak-Free Scaling    │ ──> StandardScaler fit strictly on Train, applied to Test
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 5. SMOTE Resampling     │ ──> Train Only: 227,451 Legitimate & 227,451 Fraud (1:1)
└───────────┬─────────────┘     (Test Set remains 100% natural and untouched)
            ▼
┌─────────────────────────┐
│ 6. Model Training       │ ──> Logistic Regression (Linear baseline)
└───────────┬─────────────┘ ──> Random Forest Classifier (100 trees, max_depth=10)
            ▼
┌─────────────────────────┐
│ 7. Model Evaluation     │ ──> Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
└─────────────────────────┘
```

---

## 📈 Benchmark Performance Results

Evaluated on the **untouched test set** (56,962 transactions, 98 actual frauds):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (SMOTE)** | 97.41% | 5.80% | **91.84%** | 0.1092 | 0.9699 | 0.7249 |
| **Random Forest (SMOTE)** ⭐ | **99.77%** | **42.79%** | **87.76%** | **0.5753** | **0.9836** | **0.8075** |

### 🏆 Winning Model: Random Forest Classifier
- Intercepts **87.76% of fraudulent transactions** (86/98).
- Reduces false alarms from **1,458 (Logistic Regression)** down to **115 (Random Forest)** — a **92% reduction in false positives**.
- Dominates in **PR-AUC (0.8075)** and **F1-Score (0.5753)**.

---

## 🔍 Key Notebook Sections

1. **Introduction & Business Problem:** Fraud cost analysis (False Negatives vs. False Positives).
2. **Library Imports & Plot Styling:** Publication-grade visual configuration.
3. **Dataset Ingestion & Inspection:** Schema audit, data types, missing value validation.
4. **Class Imbalance Quantification:** Exact percentage computation, Donut & Log Countplot visualizations.
5. **Exploratory Data Analysis (EDA):**
   - *5.1 Amount Distribution:* Skewness analysis, Log1p KDE, Boxplots, and ticket sizing patterns.
   - *5.2 Time-of-Day Dynamics:* Circadian diurnal rhythms, overnight vulnerability spikes, and dataset limitations.
   - *5.3 Correlation Structure:* Leading positive and negative PCA correlation drivers.
6. **Why Accuracy is Misleading:** Concrete mathematical proof demonstrating why a 99.83% accurate dummy model is completely useless.
7. **Stratified Train/Test Partitioning:** Strict 80/20 split preventing data leakage.
8. **Feature Preprocessing:** `StandardScaler` fitted strictly on training data.
9. **SMOTE Imbalance Handling:** Oversampling mechanics applied solely to training data.
10. **Model Training:** Training Logistic Regression and Random Forest.
11. **Comprehensive Model Evaluation:** Side-by-side Confusion Matrices, Classification Reports, ROC Curves, and Precision-Recall Curves.
12. **Precision vs. Recall Deep-Dive:** Financial trade-offs and multi-tiered decision thresholds.
13. **Feature Importance & Model Interpretation:** Coefficients and Gini importances with statistical caveats.
14. **Production Scalability Architecture (1M Transactions/Hour):** High-throughput event streaming (Kafka), ultra-low latency serving (<50ms SLA, ONNX/Triton), online feature store (Redis/Feast), and continuous drift monitoring.
15. **Key Findings & Actionable Recommendations:** Practical 3-tiered fraud mitigation strategies.
16. **Final Conclusion:** Submission synthesis.

---

## 🚀 How to Run the Project

### 1. Clone or Navigate to the Workspace
```bash
cd oasis_fraud_detection
```

### 2. Set Up Python Virtual Environment
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook oasis_task3_fraud_detection.ipynb
```
Select the `.venv` Python kernel and run all cells sequentially.

---

## 📁 Repository Structure
```
oasis_fraud_detection/
├── creditcard.csv                     # Kaggle benchmark dataset (284,807 rows)
├── oasis_task3_fraud_detection.ipynb  # Executed, submission-ready Jupyter Notebook
├── build_notebook.py                  # Reproducible notebook generator script
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```
