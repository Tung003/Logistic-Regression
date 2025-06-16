import pandas as pd
import os

data_path="/home/chu-tung/Desktop/machine_learning/Logistic_Regression/data/raw/framingham.csv"

def dele_nan():
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File không tồn tại: {data_path}")
    df=pd.read_csv(data_path)
    df=df.dropna()
    df.to_csv("/home/chu-tung/Desktop/machine_learning/Logistic_Regression/data/processed/data_delete_NAN.csv")

dele_nan()