import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import os
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Set up logging
def setup_logger(name, log_file='errors.log'):
    """Set up logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.ERROR)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger('model_training')


def load_params(params_path: str) -> dict:
    """Load parameters from the given path."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.info(f"Parameters loaded successfully from {params_path}")
        return params
    except FileNotFoundError as e:
        logger.error(f"File not found at {params_path}: {e}")
        raise FileNotFoundError(f"File not found at {params_path}") from e
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
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


def train_model(X, y, params):
    """
    Train LightGBM model with given parameters.
    
    Args:
        X: Feature dataframe
        y: Target variable
        params: Model parameters
        
    Returns:
        Trained model
    """
    try:
        logger.info("Training LightGBM model with parameters:")
        logger.info(f"{params}")
        
        model = LGBMRegressor(**params, random_state=42)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        logger.info(f"Training RMSE: {rmse:.4f}")
        logger.info(f"Training R²: {r2:.4f}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


def save_model(model, model_path='models'):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model object
        model_path: Directory to save the model
    """
    try:
        os.makedirs(model_path, exist_ok=True)
        model_file = os.path.join(model_path, 'lgbm_model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
            
        logger.info(f"Model saved successfully at {model_file}")
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def main():
    """Main function to run the training pipeline."""
    try:
        logger.info("Loading parameters")
        all_params = load_params('params.yaml')
        model_params = all_params.get('model_building', {})
        
        logger.info("Loading training data")
        X_train = load_data('data/feature/X_train.csv')
        y_train = load_data('data/feature/y_train.csv')
        
        # Train model
        logger.info("Training model")
        model = train_model(X_train, y_train, model_params)
        
        # Save model
        logger.info("Saving model")
        save_model(model)
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise


if __name__ == "__main__":
    main()