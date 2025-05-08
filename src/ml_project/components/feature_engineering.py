import pandas as pd
import numpy as np
import os
import pickle
import yaml
import logging
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


# Set up logging
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

train_data = pd.read_csv("data/processed/train.csv")
test_data = pd.read_csv("data/processed/test.csv")



## loading the params
def load_params(params_path: str) -> dict:
    """"Load parameters from the given path."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded successfully from {params_path}")
        return params
    except FileNotFoundError as e:
        logger.error(f"File not found at {params_path}: {e}")
        raise FileNotFoundError(f"File not found at {params_path}") from e
    except Exception as e:
        logger.error(f"Error in loading parameters: {e}")
        raise e

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from the given path.
    :param data_path: Path to the data file.
    :return: DataFrame containing the data.
    """
    try:
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        logger.info("Data loaded successfully")
        return data
    except FileNotFoundError as e:
        logger.error(f"File not found at {data_path}: {e}")
        raise FileNotFoundError(f"File not found at {data_path}") from e
    

imputer = KNNImputer(n_neighbors=5)
ss = StandardScaler()


def knnimputer(train_data , test_data):
    imputer = KNNImputer(n_neighbors=5)
    df_train_imputed = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
    df_test_imputed = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)
    return df_train_imputed , df_test_imputed



def standard_scaling(train_data , test_data):
    ss = StandardScaler()
    train_data = ss.fit_transform(train_data)
    test_data = ss.transform(test_data)

    return train_data , test_data


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    take the data from the preprocessing and after that 
    we handel the missing value of data frame
    and then apply Standard Scaler
    """
    logger.info('KNNImputer in train dataset')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    