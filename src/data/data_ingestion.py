import numpy as np
import pandas as pd
import os
import logging
import yaml
from sklearn.model_selection import train_test_split


logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_params(filepath: str) -> float:
    try:
        
        with open(filepath, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(' load params error from ' , filepath)    
        return params['data_ingestion']['test_size']  
    
    except FileNotFoundError as e:
        logger.error(f"file path not found {file_handler}, {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML error {e}")
        raise
    except Exception as e:
        logger.error(F"Unexpected error : {e}")
        raise


def load_data(filepath: str) -> pd.DataFrame:
    try:
    
        return pd.read_csv(filepath)
    
    except KeyError as e:
        logger.error(f"data not found {e}")
        raise
    except Exception as e:
        logger.error(f" error in data files {e}")
        raise


def split_data(df: pd.DataFrame, test_size: float):
    try:
       
        return train_test_split(df, test_size=test_size, random_state=42)
    except Exception as e:
        logger.error(f"split data not create {e}")

def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        logger.error(f"save data error {e} ")
        raise

def main():
    try:
        data_filepth =  r"C:/Users/Sande/Desktop/New folder (2)/notebook/Visadataset.csv"
        params_filepath = "params.yaml"
        raw_data_path = os.path.join('data', 'raw')
      
        os.makedirs(raw_data_path, exist_ok=True)
   

        df = load_data(data_filepth)
        test_size = load_params(params_filepath)
        train_data, test_data = split_data(df, test_size)

        save_data(train_data, os.path.join(raw_data_path, "train.csv"))
        save_data(test_data, os.path.join(raw_data_path, "test.csv"))
    except Exception as e:
        logger.error(f"faied to data ingestion path in error {e}")
        raise

if __name__ == "__main__":
    main()    


