import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pickle
from pathlib import Path

# This decorator tells Mage that this function is a transformer block
# It receives the output of the upstream block (the DataFrame) as its first argument
@transformer
def prepare_features(df: pd.DataFrame, *args, **kwargs):
    """
    Prepares features and target for model training.

    Args:
        df (pd.DataFrame): Input DataFrame from the data loader.
    """
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    # Convert to dictionary format for DictVectorizer
    # Only select columns that are actually used in train_dict
    train_dict = df[categorical + numerical].to_dict(orient='records')

    # Initialize and fit DictVectorizer
    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(train_dict)
    y = df['duration'].values

    # Ensure models_folder exists if not created by an exporter (good practice here)
    models_folder = Path('models')
    models_folder.mkdir(exist_ok=True)

    # Save DictVectorizer
    # Mage's @exporter could be used for this too, but for convenience within a transformer:
    dv_path = models_folder / 'DictVectorizer.b'
    with open(dv_path, 'wb') as f_out:
        pickle.dump(dv, f_out)

    print(f"Features prepared: X shape {X.shape}, y shape {y.shape}")
    print(f"DictVectorizer saved to {dv_path}")

    # Return X, y, and the DictVectorizer for the next step
    return X, y, dv