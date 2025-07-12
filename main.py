import pandas as pd
import numpy as np
from src.eda import perform_eda
from src.preprocess import preprocess_data
from src.train import stratified_split, k_fold_training
from src.model import build_autoencoder
from src.evaluate import evaluate_model
from src.interpret import explain_anomalies
from src.config import Config

# Load data
df = pd.read_csv("data/creditcard.csv")
perform_eda(df)

# Preprocessing
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=Config.test_size)

# Filter training set to only normal transactions
X_train_normal = X_train[y_train == 0]

# K-Fold CV training
def model_builder(input_dim):
    return build_autoencoder(input_dim,
                             encoding_dim=Config.encoding_dim,
                             activation=Config.activation,
                             optimizer=Config.optimizer)

model, histories = k_fold_training(X_train_normal.to_numpy(), model_builder,
                                   k=Config.k_folds,
                                   epochs=Config.epochs,
                                   batch_size=Config.batch_size)

# Evaluation
mse, threshold, y_pred = evaluate_model(model, X_test.to_numpy(), y_test.to_numpy())

# SHAP Interpretability
explain_anomalies(model, X_test.sample(100))