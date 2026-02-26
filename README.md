# Intelligent-ML-Analytics-Project

End-to-end machine learning project built on the Olist Brazilian e-commerce dataset.  
The notebook implements five production-relevant analytics modules in one workflow:

1. Regression for order value prediction
2. Classification for high-value order detection
3. Customer clustering for segmentation
4. NLP sentiment analysis on review text
5. Time-series forecasting for monthly sales

## Table of Contents

- [Project Overview](#project-overview)
- [Business Objectives](#business-objectives)
- [Repository Structure](#repository-structure)
- [Data Sources](#data-sources)
- [Feature Engineering](#feature-engineering)
- [Modeling Modules](#modeling-modules)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Outputs](#outputs)
- [Operational Notes](#operational-notes)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)

## Project Overview

This project combines predictive modeling, unsupervised learning, NLP, and forecasting in a single notebook pipeline (`mini_project.ipynb`) over transactional e-commerce data.

Core workflow:

- Load and merge orders, customers, order items, payments, reviews, and product tables
- Engineer temporal/logistics/behavioral features
- Train and evaluate models for each business problem
- Compare models with common metrics and visual diagnostics

## Business Objectives

- Predict customer order spend (`payment_value`)
- Flag high-value orders (`high_value_order`)
- Segment customers by payment, delivery, and review behavior
- Classify sentiment from Portuguese review text
- Forecast monthly sales trend for planning

## Repository Structure

```text
ML_Mini_project/
├── mini_project.ipynb
├── ML_Mini_Project_Report.docx
├── Model_Comparison_Project_table.xlsx
├── Model_Comparison_Project_table.ods
└── input/
    ├── olist_orders_dataset.csv
    ├── olist_customers_dataset.csv
    ├── olist_order_items_dataset.csv
    ├── olist_order_payments_dataset.csv
    ├── olist_order_reviews_dataset.csv
    ├── olist_products_dataset.csv
    ├── olist_geolocation_dataset.csv
    ├── olist_sellers_dataset.csv
    └── product_category_name_translation.csv
```

## Data Sources

The project uses the Olist public e-commerce dataset files in the `input/` directory.

Primary tables used in modeling:

- Orders
- Customers
- Order items
- Payments
- Reviews
- Products

## Feature Engineering

Implemented engineered features include:

- `delivery_days`
- `approval_days`
- `delivery_delay`
- `total_order_cost`
- `freight_ratio`
- `purchase_month`
- `purchase_dayofweek`

Missing-value strategy includes median imputation for numeric spend/logistics features, `-1` for incomplete time events, and empty strings for missing review text.

## Modeling Modules

### 1. Regression

- **Target:** `payment_value`
- **Models:** Linear Regression, Ridge, Random Forest Regressor, Lasso
- **Metrics:** MAE, RMSE, R2

### 2. Classification

- **Target:** `high_value_order` (1 if above median `payment_value`, else 0)
- **Models:** Logistic Regression, Decision Tree, Random Forest, KNN
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC

### 3. Clustering

- **Entity:** customer-level aggregated behavior
- **Model:** KMeans (`k=4`)
- **Selection diagnostics:** Elbow method, Silhouette score
- **Visualization:** 2D PCA projection

### 4. NLP Sentiment

- **Input:** `review_comment_message`
- **Preprocessing:** lowercasing, punctuation/number cleanup, tokenization, Portuguese stopword removal
- **Vectorization:** TF-IDF (`max_features=3000`)
- **Target label mapping:**  
  `review_score <= 2 -> Negative`, `3 -> Neutral`, `>= 4 -> Positive`
- **Classifier:** Logistic Regression
- **Metrics:** Accuracy, Precision (weighted), Recall (weighted), F1 (weighted), confusion matrix

### 5. Time Series Forecasting

- **Series:** monthly aggregated `payment_value`
- **Model:** ARIMA `(1,1,1)`
- **Validation:** holdout on final 6 months
- **Metrics:** MAE, RMSE

## Tech Stack

- Python 3.10+
- Jupyter Notebook
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- statsmodels
- nltk

## Setup

```bash
cd /home/anurag-mandlekar/Desktop/ML_Mini_project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels nltk jupyter
```

## How to Run

1. Ensure all CSV files are present under `input/`.
2. Start Jupyter:

```bash
jupyter notebook mini_project.ipynb
```

3. Run cells sequentially from top to bottom.
4. For first-time NLP execution, allow NLTK downloads (`punkt`, `stopwords`).

## Outputs

The notebook generates:

- Model comparison tables for regression and classification
- ROC curve comparisons
- Cluster profiles and segment visualizations
- Sentiment distribution and keyword insights
- 12-step monthly sales forecast and evaluation plots

Supporting report and comparison tables are available in:

- `ML_Mini_Project_Report.docx`
- `Model_Comparison_Project_table.xlsx`
- `Model_Comparison_Project_table.ods`

## Operational Notes

- Reproducibility is partially controlled with `random_state=42` across most train/test splits and models.
- Current notebook data loading uses absolute local paths. For portability, switch to relative paths (for example `input/olist_orders_dataset.csv`).

## Known Limitations

- Single-notebook workflow (limited modularity and testability)
- No experiment tracking or model registry
- No API or batch inference packaging
- No automated retraining pipeline

## Roadmap

1. Refactor notebook into reusable Python modules.
2. Add `requirements.txt`, linting, tests, and CI.
3. Add ML experiment tracking (for example MLflow).
4. Package best models for batch/API inference.
5. Add data validation and drift monitoring for ongoing production use.
