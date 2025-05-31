import pandas as pd
import numpy as np
import os
import pickle
import yaml
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer




def load_params(params_path:str) :
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


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def prepare_features(df: pd.DataFrame):
    X = df.drop(columns=['case_status'])
    y = df['case_status']
    return X, y





 
def build_pipeline(n_estimators: int, 
                   max_depth: int, 
                   min_samples_split: int, 
                   min_samples_leaf: int,
                   max_features: float,
                   class_weight: dict,
                   random_state= int
                   ) -> Pipeline:

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



def train(pipeline: Pipeline, X,y) -> Pipeline:
    pipeline.fit(X, y)
    return pipeline

def save_model(pipeline: Pipeline, output_path: str):
    with open(output_path, "wb") as file:
        pickle.dump(pipeline, file)


def main():
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


if __name__ == "__main__":
    main()


 