#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import sklearn

mlflow.set_tracking_uri('http://localhost:5000') #sqlite:///mlflow.db
mlflow.set_experiment('nyc-orchestration')

models_folder = Path('models')
models_folder.mkdir(exist_ok = True)

def download_data(year, month):
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(filename)
    print(f'there are {len(df)} records')

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

def prepare_data(df, dv = None):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    train_dict = df[categorical + numerical].to_dict(orient = 'records')

    if dv is None:
        dv = DictVectorizer(sparse = True)
        X = dv.fit_transform(train_dict)
    else:
        X = dv.transform()

    y = df['duration'].values

    return X, y, dv



def train_model(X, y, dv):
    with mlflow.start_run():
        
        mlflow.set_tag('Developer', 'OcheAI')
        model = LinearRegression()
        model.fit(X, y)

        # Get the intercept
        intercept = model.intercept_
        mlflow.log_metric("intercept", model.intercept_)
        print(f"The intercept of the Linear Regression model is: {intercept}")

        with open ('models/DictVectorizer.b', 'wb') as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(local_path='models/DictVectorizer.b', artifact_path='DictVectorizer')

        mlflow.sklearn.log_model(sk_model=model, artifact_path="lin_reg_model")

        #return model

def run_pipeline(year, month):
    df = download_data(year, month)
    X,y,dv = prepare_data(df)
    train_model(X,y,dv)

if __name__ == '__main__':
    #method 1 without specifying the arguments in cmd
    #run_pipeline(year = 2023, month = 3)

    #method 2: specifying the arguments in cmd
    import argparse

    parser = argparse.ArgumentParser(description='train a model to predict taxi trip duration')
    parser.add_argument('--year', type=int, required=True, help='year of the training data')
    parser.add_argument('--month', type=int, required=True, help='month of the training data')
    args = parser.parse_args()

    run_pipeline(year=args.year, month=args.month)
    
