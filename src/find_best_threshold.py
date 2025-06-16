from model import Logistic_Regression
from split_data import split_data
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, precision_score, recall_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# range threshold
thresholds = np.arange(0.1, 0.9, 0.01)


f1_scoresplt = []
precisionsplt = []
recallsplt = []

def train(x_train, y_train, threshold):
    model = Logistic_Regression(0.01,5000,threshold)
    model.fit(x_train, y_train)
    return model

def main_find():
    x_train, x_test, y_train, y_test = split_data()

    # SMOTE
    smote = SMOTE(random_state=42)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    # reshape y
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    # StandardScaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #best recall 
    best_recall = 0
    best_threshold_recall = 0
    #best F1
    best_f1 = 0
    best_threshold_f1=0
    #for test thresholds
    for threshold in thresholds:
        print(f"\n=== Threshold: {threshold:.2f} ===")
        model = train(x_train, y_train, threshold)
        y_pred = model.predict(x_test)

        # Metrics
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        acc = model.accuracy(x_test, y_test)

        # save
        f1_scoresplt.append(f1)
        recallsplt.append(recall)
        precisionsplt.append(precision)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold_f1 = threshold

        if recall > best_recall and f1>0.35:
            best_recall = recall
            best_threshold_recall = threshold

    # map
    plt.figure(figsize=(10, 6))
    plt.scatter(best_threshold_recall,  best_recall, color='green', s=100, label='Best Recall')
    plt.annotate(f"({best_threshold_recall:.2f}, {best_recall:.2f})", 
             (best_threshold_recall, best_recall), 
             textcoords="offset points", 
             xytext=(0,10), ha='center', color='green')
    plt.scatter(best_threshold_f1,      best_f1,     color='red',   s=100, label='Best F1-score')
    plt.annotate(f"({best_threshold_f1:.2f}, {best_f1:.2f})", 
             (best_threshold_f1, best_f1), 
             textcoords="offset points", 
             xytext=(0,10), ha='center', color='red')
    plt.plot(thresholds, f1_scoresplt,  label="F1-score",   marker='o')
    plt.plot(thresholds, precisionsplt, label="Precision",  linestyle='--')
    plt.plot(thresholds, recallsplt,    label="Recall",     linestyle=':')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold test")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\nThreshold best F1-score: {best_threshold_f1:.2f}   with F1 = {best_f1:.3f}")
    print(f"\nThreshold best Recall: {best_threshold_recall:.2f} with Recall = {best_recall:.3f}")

if __name__ == "__main__":
    main_find()
