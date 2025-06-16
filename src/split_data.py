import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

data_path="/home/chu-tung/Desktop/machine_learning/Logistic_Regression/data/processed/data_delete_NAN.csv"
def split_data():
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File không tồn tại: {data_path}")
    df=pd.read_csv(data_path)
    target="TenYearCHD"
    x=df.drop(target,axis=1)
    y=df[target]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    # print("shape of x train: ", x_train.shape)
    # print("shape of y train: ", y_train.shape)
    return x_train,x_test,y_train,y_test
