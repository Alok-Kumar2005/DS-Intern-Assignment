import numpy as np
import pandas as pd
import os
import logging
import yaml
from sklearn.preprocessing import StandardScaler

# Set up logging
logger = logging.getLogger('data_preprocessing')
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



## to handle the outliers
def clamp_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Winsorize numeric columns so that any values below
    Q1 - 1.5*IQR become exactly Q1 - 1.5*IQR, and likewise
    for values above Q3 + 1.5*IQR.
    """
    df_clamped = df.copy()
    num_cols = df_clamped.select_dtypes(include="number").columns

    for col in num_cols:
        Q1, Q3 = df_clamped[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR

        df_clamped[col] = df_clamped[col].clip(lower=lower, upper=upper)

    return df_clamped


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data removing the column with less than 0.01 threshold 
    value and taking average of random variable column,
    dropping all columns from index 21 to the second-to-last column.
    
    :param df: DataFrame containing the data.
    :return: DataFrame containing the preprocessed data.
    """
    try:
        logger.info('Dropping column having threshold less than 0.01 with target column')
        drop = ['zone2_humidity', 'zone3_humidity', 'zone4_humidity',
                'zone5_temperature', 'zone5_humidity', 'zone7_temperature',
                'zone7_humidity', 'zone9_temperature', 'visibility_index',
                'dew_point', 'minute']
        df = df.drop(columns=drop)
        logger.info('taking average of random_varaible column')
        df['random_variable'] = (df['random_variable1'] + df['random_variable2'])/2
        df = df.drop(columns=['random_variable1', 'random_variable2'])
        
        return df
    except Exception as e:
        logger.error(f"Error in preprocessing data: {e}")
        raise e


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str):
    """
    Save the train and test data to the given path.
    :param train_data: DataFrame containing the training data.
    :param test_data: DataFrame containing the testing data.
    :param data_path: Path to save the data.
    """
    try:
        logger.info(f"Saving train and test data to {data_path}")
        os.makedirs(data_path, exist_ok=True)
        train_file_path = os.path.join(data_path, "train.csv")
        test_file_path = os.path.join(data_path, "test.csv")
        train_data.to_csv(train_file_path, index=False)
        test_data.to_csv(test_file_path, index=False)
        logger.info("Data saved successfully")
    except Exception as e:
        logger.error(f"Error in saving data: {e}")
        raise e
    

def main():
    try:
        df_train = load_data("data/raw/train.csv")
        df_test = load_data('data/raw/test.csv')
        df_train = preprocess_data(df_train)
        final_df_train = clamp_outliers(df_train)
        final_df_test = preprocess_data(df_test)

        save_data(final_df_train, final_df_test, "data/preprocessed")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise e


if __name__ == "__main__":
    main()