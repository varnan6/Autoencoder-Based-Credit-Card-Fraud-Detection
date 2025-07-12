# train.py
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

def stratified_split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

def k_fold_training(X_train, model_builder, k=5, epochs=10, batch_size=256):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    histories = []
    for train_idx, val_idx in skf.split(X_train, np.zeros(len(X_train))):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        model = model_builder(X_train.shape[1])
        history = model.fit(X_tr, X_tr,
                            validation_data=(X_val, X_val),
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=True,
                            verbose=0)
        histories.append(history)
    return model, histories