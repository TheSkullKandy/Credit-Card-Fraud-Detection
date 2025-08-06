# Credit Card Fraud Detection: TabPFN vs XGBoost

This repository presents a comprehensive comparison between two machine learning models — XGBoost and TabPFN — on the Credit Card Fraud Detection dataset. The goal is to evaluate their performance on both a small balanced subset and the full imbalanced dataset, addressing real-world challenges like data scarcity, class imbalance, and model interpretability.

## Project Structure

- `notebooks/`
  - `small_dataset_comparison.ipynb`: Comparison on a 500-row balanced dataset (200 fraud, 300 non-fraud).
  - `full_dataset_comparison.ipynb`: Evaluation on the full dataset (284,807 transactions).
- `data/`
  - `creditcard_sample_500_balanced.csv`: Pre-sampled 500-row dataset.
  - `creditcard.csv`: Must be downloaded manually from Kaggle (see below).
- `README.md`: Project overview and usage guide.

## Dataset

The dataset used in this project is available from Kaggle:

**Download here:**  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

It contains 284,807 credit card transactions made by European cardholders, with only 492 labeled as fraudulent (approximately 0.17%).  
Each transaction includes 30 features:
- 28 anonymized principal components (V1 to V28)
- Time
- Amount

The target column `Class` indicates whether the transaction is fraud (`1`) or not (`0`). The dataset is highly imbalanced, making it suitable for testing anomaly and fraud detection systems.

## Objectives

- Compare XGBoost and TabPFN on:
  - A small, balanced dataset (data-scarce scenario)
  - The full, imbalanced dataset (real-world scenario)
- Apply SMOTE to handle class imbalance
- Evaluate models using metrics:
  - F1 Score
  - Precision
  - Recall
  - ROC AUC
  - Mean Absolute Error (MAE)
- Visualize results:
  - Confusion matrices
  - ROC and Precision-Recall curves
  - MAE comparisons

## Dependencies

Install the required Python packages using pip:

```bash
pip install tabpfn xgboost shap imbalanced-learn scikit-learn pandas matplotlib seaborn
