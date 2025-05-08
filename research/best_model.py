import os
os.environ["DAGSHUB_DISABLE_SSL_VERIFY"] = "true"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import dagshub

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

dagshub.init(repo_owner='ay747283', repo_name='DS-Intern-Assignment', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/ay747283/DS-Intern-Assignment.mlflow')
mlflow.set_experiment('Regressor models')

# Load dataset
X_train = pd.read_csv('research/data/X_train.csv')
X_test = pd.read_csv('research/data/X_test.csv')
y_train = pd.read_csv('research/data/y_train.csv')
y_test = pd.read_csv('research/data/y_test.csv')

print(X_train.shape)

models = {
    "GradientBoostingRegressor": GradientBoostingRegressor(),
    "XGBRegressor": XGBRegressor(),
    "LGBMRegressor": LGBMRegressor(),
    "BaggingRegressor": BaggingRegressor()
}

# Dict to store performance metrics
results = {}

# Train and evaluate each model
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Log model parameters
        for param_name, param_value in model.get_params().items():
            mlflow.log_param(param_name, param_value)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Log metrics
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("RMSE", rmse)
        
        # Log model
        mlflow.sklearn.log_model(model, name)
        
        # Store results
        results[name] = {
            "MSE": mse,
            "MAE": mae,
            "R2": r2,
            "RMSE": rmse
        }