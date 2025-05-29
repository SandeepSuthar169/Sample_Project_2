import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer



train_data = pd.read_csv("./data/processed/train_preprocessing.csv")

X_train = train_data.drop(columns=['case_status'])
y_train = train_data['case_status']


process = ColumnTransformer(transformers=[
    ('one', OneHotEncoder(), ['full_time_position', 'region_of_employment', 'requires_job_training', 'has_job_experience', 'continent']),
    ('std', StandardScaler(), ['no_of_employees', 'yr_of_estab', 'prevailing_wage'])
],
    remainder="passthrough"
)

pipeline = Pipeline(steps=[
    ('process', process),
])

pipeline = Pipeline(steps =[
    ('process', process),
    ("classi", RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42))
])

pipeline.fit(X_train, y_train)

pickle.dump(pipeline, open('model.pkl', "wb"))
