
# evaluate.py
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_model(model, X_test, y_test, threshold=None):
    reconstructions = model.predict(X_test)
    mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
    if threshold is None:
        threshold = np.percentile(mse, 95)
    y_pred = (mse > threshold).astype(int)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("AUC Score:", roc_auc_score(y_test, mse))

    return mse, threshold, y_pred