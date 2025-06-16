from ydata_profiling import ProfileReport
import pandas as pd
import os

data_path="/home/chu-tung/Desktop/machine_learning/Logistic_Regression/data/raw/framingham.csv"

def data_report():
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File không tồn tại: {data_path}")
    df=pd.read_csv(data_path)
    profile = ProfileReport(df, title="Profiling Report",explorative=True)
    profile.to_file("/home/chu-tung/Desktop/machine_learning/Logistic_Regression/data/raw/framingham_report.html")

data_report()