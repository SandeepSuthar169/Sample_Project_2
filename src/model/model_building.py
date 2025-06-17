import pandas as pd
import numpy as np
import os
import pickle
import yaml
import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer

logger = logging.getLogger("model_building")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('error.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


#Exception, valueerror, filenotfounderror, attributeerrror

def load_params(params_path:str) :
    try:
        with open(params_path, 'r' ) as file:
            params = yaml.safe_load(file)
            n_estimators=params['model_building']['n_estimators'] 
            max_depth = params['model_building']['max_depth']
            min_samples_split=params['model_building']['min_samples_split'] 
            min_samples_leaf = params['model_building']['min_samples_leaf'] 
            max_features = params['model_building']['max_features']
            class_weight = params['model_building']['class_weight']  
            random_state = params['model_building']['random_state']  
            print("Loaded params:", params)   
            return n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features, class_weight, random_state
    
    except FileNotFoundError as e:
        logger.error(f"params file not found {params_path} : {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"yaml error {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error {e}")
        raise



def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError as e:
        logger.error(f"filepath not found {filepath} : {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpacted error in data loading {filepath} : {e}")
        raise


def prepare_features(df: pd.DataFrame):
    try:
        X = df.drop(columns=['case_status'])
        y = df['case_status']
        return X, y
    except Exception as e:
        logger.error(f"error occur in prepare featueres : {e}")




 
def build_pipeline(n_estimators: int, 
                   max_depth: int, 
                   min_samples_split: int, 
                   min_samples_leaf: int,
                   max_features: float,
                   class_weight: dict,
                   random_state= int
                   ) -> Pipeline:
    try:
        process = ColumnTransformer(transformers=[
            ('one', OneHotEncoder(), ['full_time_position', 'region_of_employment', 'requires_job_training', 'has_job_experience', 'continent']),
            ('std', StandardScaler(), ['no_of_employees', 'yr_of_estab', 'prevailing_wage'])
        ],
            remainder="passthrough"
        )

          
        pipeline = Pipeline(steps =[
            ('process', process),
            ("classi", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight,
                random_state=random_state))
        ])
        return pipeline
    except AttributeError as e:
        logger.error(f"error occur in pipeline {e} ")
        raise
    except Exception as e:
        logger.error(f"Unexpected error {e}")
        raise


def train(pipeline: Pipeline, X,y) -> Pipeline:
    try:
        pipeline.fit(X, y)
        return pipeline
    except Exception as e:
        logger.error(f"pipeline fit into X, y error occur {pipeline} : {e}")
        raise

def save_model(pipeline: Pipeline, output_path: str):
    try:
        with open(output_path, "wb") as file:
            pickle.dump(pipeline, file)
    except Exception as e:
        logger.error(f"error in save model {output_path} : {e}")
        raise


def main():
    try:

        input_path = './data/processed/train_processed.csv'
        model_path = 'model.pkl'
        params_path = 'params.yaml'

        n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, class_weight, random_state = load_params(params_path)
        df = load_data(input_path)
        X_train, y_train = prepare_features(df)

        Pipeline = build_pipeline(n_estimators, 
                                max_depth,
                                min_samples_leaf,
                                min_samples_split,
                                max_features,
                                class_weight,
                                random_state)
        
        trained_pipeline = train(Pipeline, X_train, y_train)
        save_model(trained_pipeline, model_path)
    except FileNotFoundError as e:
        logger.error(f"input, model, params path not found {e}")
        raise
    except AttributeError as e:
        logger.error(f"error occur in df and X_train, y_train features {e}")
        raise
    except Exception as e:
        logger.error(f"unexprected error {e}")
        raise


if __name__ == "__main__":
    main()


 
