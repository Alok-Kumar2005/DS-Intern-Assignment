import pandas as pd
import numpy as np
import os
import pickle
import yaml
import logging
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


# Set up logging
logger = logging.getLogger('feature engineering')
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
    



def knnimputer(train_data , test_data):
    imputer = KNNImputer(n_neighbors=5)
    df_train_imputed = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
    df_test_imputed = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)
    return df_train_imputed , df_test_imputed



def standard_scaling(train_data , test_data):
    ss = StandardScaler()
    train_data_scaled = ss.fit_transform(train_data)
    test_data_scaled = ss.transform(test_data)
    
    # Save the scaler model
    os.makedirs('data/models', exist_ok=True)
    with open('data/models/standard_scaler.pkl', 'wb') as f:
        pickle.dump(ss, f)
    logger.info("StandardScaler model saved successfully")
    
    return train_data_scaled, test_data_scaled

def save_data(data, data_path: str, filename: str):
    """
    Save data to the given path.
    :param data: DataFrame containing the data.
    :param data_path: Path to save the data.
    :param filename: Name of the file.
    """
    try:
        logger.info(f"Saving data to {data_path}")
        os.makedirs(data_path, exist_ok=True)
        file_path = os.path.join(data_path, filename)
        
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data.to_csv(file_path, index=False)
        else:
            pd.DataFrame(data).to_csv(file_path, index=False)
            
        logger.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error in saving data: {e}")
        raise e





def main():
    try:
        df_train = load_data("data/preprocessed/train.csv")
        df_test = load_data('data/preprocessed/test.csv')  
        
        logger.info('Knn Imputer on test and train data')
        df_train_imputed, df_test_imputed = knnimputer(df_train, df_test)
        
        logger.info('Extracting target variable')
        y_train = df_train_imputed['equipment_energy_consumption']
        y_test = df_test_imputed['equipment_energy_consumption']
        
        X_train_imputed = df_train_imputed.drop(columns=['equipment_energy_consumption']) 
        X_test_imputed = df_test_imputed.drop(columns=['equipment_energy_consumption'])    
        
        logger.info('Standard Scaler in dataset')
        X_train_scaled, X_test_scaled = standard_scaling(X_train_imputed, X_test_imputed)

        logger.info('Saving new data to the data folder ')
        os.makedirs("data/feature", exist_ok=True)
        save_data(X_train_scaled, "data/feature", "X_train.csv")
        save_data(X_test_scaled, "data/feature", "X_test.csv")
        save_data(y_train, "data/feature", "y_train.csv")
        save_data(y_test, "data/feature", "y_test.csv")
        
        logger.info("Data processing completed successfully")
        print(y_train.shape)
        print(y_test.shape)
        print(X_train_scaled.shape)
        print(X_test_scaled.shape)        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise e


if __name__ == "__main__":
    main()

