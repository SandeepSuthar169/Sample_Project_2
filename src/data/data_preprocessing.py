import numpy as np
import pandas as pd
import os
import logging

logger = logging.getLogger('data preprocessing')
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




def load_data(filepath:str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath) 
    except FileNotFoundError as e:
        logger.error(f"data file not found error {filepath}: {e} ")
        raise
    except Exception as e:
        logger.error(f"Unexpected error of data loading : {e}")

    

def columns_label_encoding(df):
    try:
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
    
    except Exception as e:
        logger.error(f"columns label encoding occurs error : {e}")
        raise

def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index= False)
    except ValueError as e:
        logger.error(f"saveing data into DataFrame occurs error {pd.DataFrame} : {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error of data saveing : {filepath} : {e}")
        raise


def main():
    try:
        raw_data_path = "./data/raw"
        processed_data_path = "./data/processed"

        train_data = load_data(os.path.join(raw_data_path, "train.csv"))
        test_data =load_data(os.path.join(raw_data_path, "test.csv"))

        train_preprocessing_data = columns_label_encoding(train_data)
        test_preprocessing_data = columns_label_encoding(test_data)

        os.makedirs(processed_data_path)

        save_data(train_preprocessing_data, os.path.join(processed_data_path,'train_processed.csv'))
        save_data(test_preprocessing_data, os.path.join(processed_data_path,  'test_processed.csv'))
    except FileNotFoundError as e:
        logger.error(f"raw data, processed data not found {e}")
        raise 
    except AttributeError as e:
        logger.error(f"error occur in train_data, test_data, train_preprocessing_data, test_preprocessing_data files :  {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected errro of data main files {e}")
        raise
if __name__ == "__main__":
    main()









