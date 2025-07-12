# preprocess.py
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df = df.copy()
    scaler = StandardScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y
