import numpy as np
import pandas as pd
import os

train_data = pd.read_csv('./data/raw/train.csv')
test_data = pd.read_csv('./data/raw/test.csv')


def drop_columns(df: pd.DataFrame):
    df = df.dop('case_id', axis= 1)
    return df


def columns_label_encoding(df):
    df['education_of_employee'] = df['education_of_employee'].replace({
    "Bachelor's" : 1,
    "Master's": 2,
    "High School": 3,
    "Doctorate": 4
    })
    df['education_of_employee'] = df['education_of_employee'].astype(int)

    df['unit_of_wage'] = df['unit_of_wage'].replace({
    "Year": 4,            
    "Hour": 1,      
    "Week": 2,       
    "Month" :3
    })
    df['unit_of_wage'] = df['unit_of_wage'].astype(int)

    df['case_status'] = df['case_status'].replace({
    "Certified": 1,
    "Denied": 0
    })
    df['case_status'] = df['case_status'].astype(int)
    return df


train_preprocessing_data = columns_label_encoding(train_data)
test_preprocessing_data = columns_label_encoding(test_data)

data_path = os.path.join("data", "processed")
os.makedirs(data_path)

train_preprocessing_data.to_csv(os.path.join(data_path, "train_preprocessing.csv" ))
test_preprocessing_data.to_csv(os.path.join(data_path, "test_preprocessing.csv"))