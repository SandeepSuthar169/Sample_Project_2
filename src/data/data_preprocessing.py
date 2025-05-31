import numpy as np
import pandas as pd
import os




def load_data(filepath:str) -> pd.DataFrame:
    return pd.read_csv(filepath) 


# def drop_columns(df: pd.DataFrame):
#     df = df.drop('case_id', axis= 1)
#     return df



def columns_label_encoding(df):
    df=df.drop('case_id', axis = 1)


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

def save_data(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index= False)


def main():
    raw_data_path = "./data/raw"
    processed_data_path = "./data/processed"

    train_data = load_data(os.path.join(raw_data_path, "train.csv"))
    test_data =load_data(os.path.join(raw_data_path, "test.csv"))

    train_preprocessing_data = columns_label_encoding(train_data)
    test_preprocessing_data = columns_label_encoding(test_data)

    os.makedirs(processed_data_path)

    save_data(train_preprocessing_data, os.path.join(processed_data_path,'train_processed.csv'))
    save_data(test_preprocessing_data, os.path.join(processed_data_path,  'test_processed.csv'))

if __name__ == "__main__":
    main()









