import numpy as np
import pandas as pd
import os
import logging
import yaml
from sklearn.model_selection import train_test_split


def load_params(filepath: str) -> float:
    with open(filepath, 'r') as file:
        params = yaml.safe_load(file)
    return params['data_ingestion']['test_size']  


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def split_data(df: pd.DataFrame, test_size: float):
    return train_test_split(df, test_size=test_size, random_state=42)

def save_data(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=False)


def main():
    data_filepth =  r"C:/Users/Sande/Desktop/New folder (2)/notebook/Visadataset.csv"
    params_filepath = "params.yaml"
    raw_data_path = os.path.join('data', 'raw')

    os.makedirs(raw_data_path, exist_ok=True)

    df = load_data(data_filepth)
    test_size = load_params(params_filepath)
    train_data, test_data = split_data(df, test_size)

    save_data(train_data, os.path.join(raw_data_path, "train.csv"))
    save_data(test_data, os.path.join(raw_data_path, "test.csv"))

if __name__ == "__main__":
    main()    



# df = pd.read_csv(r"C:/Users/Sande/Desktop/New folder (2)/notebook/Visadataset.csv")
# df = df.drop('case_id', axis=1)

# # Split data
# train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# # Save split data
# data_path = os.path.join('data', 'raw')
# os.makedirs(data_path, exist_ok=True)

# train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
# test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)


