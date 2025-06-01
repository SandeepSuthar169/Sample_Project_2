import pandas as pd
import json
import os
import pickle
import logging
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

logger = logging.getLogger("model_evaluation")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('error.log')
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_test_data(filepath: str)-> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except KeyError as e:
        logger.error(f'load test data not found {filepath} : {e}')
        raise
    except Exception as e:
        logger.error(f"Unexpected error in load data {e}")
        raise

def load_model(model_path: str):
    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError as e:
        logger.error(f"pickle file not found occur error {model_path} : {e}")
        raise

def evaluate_model(model, X_test, y_test) -> dict:
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        return accuracy, f1, precision, recall
    except AttributeError as e:
        logger.error(f"prediction error occur {model} :  {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occur {e} ")
        raise


def save_metrics(metrics: dict, output_path:str):
    try:
        with open(output_path, 'w') as file:
            json.dump(metrics, file, indent=4)
    except Exception as e:
        logger.error(f"metrics.josm error occur {e}")
        raise
    

def main():
    try:
            
        test_data_path = "./data/processed/test_processed.csv"
        model_path = 'model.pkl'
        metrics_path = 'metrics.json'

        test_data = load_test_data(test_data_path)
        
        X_test = test_data.drop(columns=['case_status'])
        y_test = test_data['case_status']

        model = load_model(model_path)
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, metrics_path)

    except FileNotFoundError as e:
        logger.error(f"test, modela-path, metrics-path file not found {e}")
        raise
    except AttributeError as e:
        logger.error(f"loading data error as {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occur {e}")
        raise

if __name__ == "__main__":
    main()
