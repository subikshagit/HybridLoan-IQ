# Loan Approval Prediction (HybridLoan IQ)

Machine Learning Pipeline in Production

This project builds a loan approval prediction system using a classic machine learning workflow (data cleaning feature engineering  model training evaluation) and exposes predictions through a simple Gradio UI.

> Notebook: `LoanApproval (1).ipynb`


## What's Inside

- **Data loading & inspection**: loads a loan approval dataset (CSV), checks shape, dtypes, nulls, and basic stats.
- **Cleaning**:
  - Drops `loan_id`
  - Strips column names
  - Fixes negative values in `residential_assets_value` by replacing them with the median
- **Encoding**: label-encodes categorical columns (e.g., `education`, `self_employed`, `loan_status`).
- **EDA**: histograms, skewness checks, correlation heatmap.
- **Skew handling**: applies Yeo-Johnson power transform to selected skewed features.
- **Class imbalance**: applies **SMOTE** to balance `loan_status`.
- **Feature selection**: uses `SelectKBest(f_classif)` to inspect top features.
- **Feature engineering**:
  - `loan_to_income = loan_amount / income_annum`
  - `total_asset = residential_assets_value + commercial_assets_value`
  - `EMI = loan_amount / loan_term`
- **Scaling**: uses `MinMaxScaler` on numeric columns.
- **Model training**: trains **Logistic Regression**, evaluates with accuracy, classification report, confusion matrix, ROC curve.
- **Threshold tuning**: optional custom probability threshold (example: `0.55`).
- **Model export**: saves a model with `joblib`.
- **UI**: Gradio interface that predicts loan approval and (optionally) generates an explanation using a Generative AI model.

## How to Run (Local)


### 1) Create environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib joblib gradio google-generativeai
```

### 3) Run the notebook

Open `LoanApproval (1).ipynb` in VS Code (Jupyter) and run cells top-to-bottom.

> Note: The first cells in the notebook use **Google Colab** drive mounting. For local runs, replace the Colab path with a local CSV path.

## Model & Inference Notes

- The prediction function constructs engineered features (`loan_to_income`, `total_asset`, `EMI`) and then calls `model.predict(...)`.
- For best correctness, the **feature order** at inference time must exactly match the order used during training.

## Security / API Keys

If you use `google-generativeai`, do **not** hard-code API keys in notebooks or code.

Prefer using an environment variable, e.g.:

```bash
setx GOOGLE_API_KEY "your_key_here"
```

…and then in Python:

```python
import os
api_key = os.getenv("GOOGLE_API_KEY")
```

## Image

Place the provided diagram image into:

- `assets/ml-pipeline-in-production.png`

The README will render it automatically once the file exists.

## Outputs

- `loan_model.pkl`  trained model artifact saved via `joblib` (location depends on where you run the notebook).
