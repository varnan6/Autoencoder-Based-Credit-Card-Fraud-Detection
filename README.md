# Social Media vs Productivity — ML Pipeline
---
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10.4%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-blue)
![Last Commit](https://img.shields.io/github/last-commit/varnan6/Autoencoder-Based-Credit-Card-Fraud-Detection)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-available-orange)
---
This project analyzes how various lifestyle and behavioral factors — including social media usage — impact actual productivity scores. It performs exploratory data analysis, builds preprocessing and regression pipelines, performs model tuning via Grid Search, and interprets results using SHAP.

## Project Structure

---

```bash
creditcard_autoencoder/
│   main.py
│   README.md
│
├─── /data
│       creditcard.csv
│
├─── /src
│       config.py
│       eda.py
│       evaluate.py
│       interpret.py
│       model.py
│       preprocess.py
│       train.py
│
└─── /notebooks
        exploration.ipynb
```

---

## Features Used

From the original dataset, the following columns are selected and processed:

- **PCA-Transformed Features**: `V1` to `V28`
- **Original Features**: `Time`, `Amount`
- **Target**: `Class` (`0` = Normal, `1` = Fraudulent)

<img src = "images/heatmap.png"/>

Transactions are normalized and fraud is detected by analyzing reconstruction error using an unsupervised autoencoder.

---

## Workflow

### 1. Data Preprocessing (`preprocess.py`)
- `Time` and `Amount` are scaled with `StandardScaler`.
- Data is splot into train/test with stratification on `class`.
- Only normal transactions are used to train the autoencoder.

### 2. Exploratory Data Analysis (`eda.py`)
- Class imbalance visualization.
- Feature distribution histograms.
- Correlation heatmaps.
- PCA/t-SNE visualization of transaction clusters.

### 3. Autoencoder Model (`model.py`)
- Trained to minimize reconstruction loss (MSE).
- Tuned using parameters like encoding dimension, batch size, epochs, etc.

### 4. Training with Cross-Validation (`train.py`)
- Used 5-fold Stratified K-Fold Cross Validation.
- Trains on Class 0 (normal) data only.
- Final model selected and evaluated on the full test set.

### 5. Evaluation (`evaluate.py`)
- Predicts reconstruction error for test transactions.
- Flags frauds based on a threshold.
- Metrics: Confusion Matrix, Precision, Recall, F1, AUC

### 6. Explainability (`interpret.py`)
- Uses SHAP to identify which features contribute most to reconstruction error.
- Helps interpret model behavior for both normal and fraud predictions.

---

## Requirements

### Installing dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the project

```bash
python main.py
```

The pipeline will:
1) Load and explore the data
2) Preprocess features and scale them
3) Train teh autoencoder with K-Fold Cross Validation
4) Evaluate on test set with anomaly scoring
5) Generate interpretability plots with SHAP

---

## Current Output

Results: 
<img src="images/results.png"></img>

Best parameters:
<img src="images/params.png"></img>

---

## Dataset

The dataset used sources from [Machine Learning Group - ULB](https://www.kaggle.com/organizations/mlg-ulb) [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) Kaggle dataset website.

---

## Future improvements

- Compare autoencoder with isolation forests, One-Class SVM
- Hyperparameter tuning via `KerasTuner`
- Deploy API for fraud scoring
- Real-time stream integration with Kafka or Spark
- Web dashboard for fraud alerts


---

## License

This project is licensed under the [MIT License](LICENSE).
