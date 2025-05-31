import pandas as pd
import json
import os
import pickle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

test_data = pd.read_csv("./data/processed/test_processed.csv")

X_test = test_data.drop(columns=['case_status'])
y_test = test_data['case_status']

pipeline = pickle.load(open('model.pkl', "rb"))

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

metrics= {
    'accuracy': accuracy
}

with open('metrics.json', 'w') as file:
    json.dump(metrics, file, indent=4)