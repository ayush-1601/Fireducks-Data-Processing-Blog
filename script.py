import pandas as pd
import os
from time import time
from functools import wraps

# Data Preparation

dir = "data_sets"
df_list = []
for year in [2022, 2023]:
    for month in range(1, 13):
        month_str = str(month).zfill(2)
        file_path = os.path.join(dir, f"yellow_tripdata_{year}-{month_str}.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            df_list.append(df)

taxi_data = pd.concat(df_list)
taxi_data.to_csv("taxi_all.csv", index=False)

# Performance Measurement Wrapper
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} : {time() - start_time:.5f} sec")
        return result
    return wrapper

# Data Processing Functions

@timer
def load_data(filename):
    df = pd.read_csv(filename)
    return df

@timer
def drop_na(df):
    df.dropna(how="all", inplace=True)
    return df

@timer
def convert_to_datetime(df, column):
    df[column] = pd.to_datetime(df[column])
    return df

@timer
def filter_passengers(df):
    return df[df["passenger_count"] > 0]

@timer
def extract_date_time_features(df):
    df['year'] = df['tpep_pickup_datetime'].dt.year
    df['month'] = df['tpep_pickup_datetime'].dt.month
    df['day'] = df['tpep_pickup_datetime'].dt.day
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    return df

@timer
def compute_trip_duration(df):
    df["trip_duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.seconds / 60
    return df[(df["trip_duration"] > 0) & (df["trip_duration"] <= 180)]

# FireDucks Benchmarking
def evaluate(df):
    try:
        df._evaluate()  # Force evaluation for FireDucks
    except AttributeError:
        pass

if __name__ == "__main__":
    filename = "taxi_all.csv"
    df = load_data(filename)
    df = drop_na(df)
    df = convert_to_datetime(df, "tpep_pickup_datetime")
    df = convert_to_datetime(df, "tpep_dropoff_datetime")
    df = filter_passengers(df)
    df = extract_date_time_features(df)
    df = compute_trip_duration(df)
    evaluate(df)