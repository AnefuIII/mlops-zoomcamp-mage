import pandas as pd
from pathlib import Path
from mage_ai.data_preparation.decorators import data_loader

# This decorator tells Mage that this function is a data loader block
# The **kwargs allows you to pass pipeline parameters like 'year' and 'month'
@data_loader
def load_trip_data(year: int, month: int, *args, **kwargs) -> pd.DataFrame:
    """
    Downloads and performs initial cleaning of yellow taxi trip data.

    Args:
        year (int): The year of the data to download.
        month (int): The month of the data to download.
    """
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    print(f'Downloading data for {year}-{month:02d}...')
    df = pd.read_parquet(filename)
    print(f'There are {len(df)} records initially.')

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    # Filter durations
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    print(f'Filtered to {len(df)} records after duration cleaning.')

    # Convert categorical IDs to string
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df