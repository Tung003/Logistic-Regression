from model import Logistic_Regression
from split_data import split_data
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, precision_score, recall_score,classification_report,confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def train(x_train,y_train):
    #model
    model=Logistic_Regression(learing_rate=0.01,num_iterations=5000,threshold=0.46)
    model.fit(x_train,y_train)
    return model

#test with framingham dataset
def main():
    x_train,x_test,y_train,y_test=split_data()
    #balanced train data
    smote = SMOTE(random_state=42)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    # print("shape: ",x_test.shape,y_test.shape)
    y_train=np.array([y_train]).reshape(-1,1)
    y_test =np.array([y_test]).reshape(-1,1)

    #StandardScaler
    scaler=StandardScaler()

    #scaler
    x_train=scaler.fit_transform(x_train)
    x_test =scaler.transform(x_test)

    #train
    model=train(x_train,y_train)
    # accuaracy
    accuaracy=model.accuracy(x_test,y_test)
    print("Model accuracy: ","%.3f"%accuaracy)
    #model predict y_test from input x_test
    y_test_pred=model.predict(x_test)
    #metrics evaluation
    f1_scores=f1_score(y_test,y_test_pred)
    recall_scores=recall_score(y_test,y_test_pred)
    precision_scores=precision_score(y_test,y_test_pred)

    print("F1_score  = ","%.3f"%f1_scores)
    print("Recall    = ","%.3f"%recall_scores)
    print("Precision = ","%.3f"%precision_scores)

    print("-----------------Classification report-----------------\n",classification_report(y_test,y_test_pred))
    #confusion_matrix
    confusion_matrix_result = confusion_matrix(y_test, y_test_pred)
    print("confusion_matrix \n",confusion_matrix_result)

    #plot confusion_matrix
    df_cm = pd.DataFrame(confusion_matrix_result)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("/home/chu-tung/Desktop/machine_learning/Logistic_Regression/outputs/metrics_refined_f1.png")
    plt.show()
    # return model.w, model.b

if __name__=="__main__":
    main()
    # with open("/home/chu-tung/Desktop/machine_learning/Logistic_Regression/models/weights_recall_B.txt","w") as f:
    #     f.write("Weights (w):\n")
    #     for i, value in enumerate(w):
    #         f.write(f"w[{i}] = {value[0]}\n")
    #     f.write("\nBias (b):\n")
    #     f.write(f"b = {b}\n")
