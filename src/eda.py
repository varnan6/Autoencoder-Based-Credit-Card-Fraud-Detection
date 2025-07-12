# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    print("\nBasic Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nClass Distribution:")
    print(df['Class'].value_counts())

    sns.countplot(x='Class', data=df)
    plt.title("Class Distribution")
    plt.show()

    corr = df.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()