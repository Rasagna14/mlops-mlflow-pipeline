import pandas as pd

def load_csv(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.fillna(0)
    return df
