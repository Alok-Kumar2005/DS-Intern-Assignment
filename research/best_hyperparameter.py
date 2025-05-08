import os
os.environ["DAGSHUB_DISABLE_SSL_VERIFY"] = "true"

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import dagshub
from lightgbm import LGBMRegressor

dagshub.init(repo_owner='ay747283', repo_name='DS-Intern-Assignment', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/ay747283/DS-Intern-Assignment.mlflow')
mlflow.set_experiment('LGBM-Hyperparameter-Tuning')

X_train = pd.read_csv('research/data/X_train.csv')
X_test = pd.read_csv('research/data/X_test.csv')
y_train = pd.read_csv('research/data/y_train.csv')
y_test = pd.read_csv('research/data/y_test.csv')

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [-1, 5, 10, 15, 20],
    'num_leaves': [20, 31, 50, 70],
    'min_child_samples': [5, 10, 20, 50]
}

# Number of random hyperparameter combinations to try
n_iter = 20

# Cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Create the base model
base_model = LGBMRegressor(random_state=42)

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=n_iter,
    scoring='neg_mean_squared_error',
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

with mlflow.start_run(run_name="LGBM-Hyperparameter-Tuning"):
    mlflow.log_param("parameter_space", str(param_grid))
    mlflow.log_param("n_iter", n_iter)
    mlflow.log_param("cv_folds", cv.n_splits)
    
    print("Starting hyperparameter tuning...")
    random_search.fit(X_train, y_train)
    
    # Get the best parameters and score
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print(f"Best RMSE: {np.sqrt(-best_score)}")
    print(f"Best parameters: {best_params}")
    
    # Log the best parameters
    for param_name, param_value in best_params.items():
        mlflow.log_param(f"best_{param_name}", param_value)
    
    # Log the best CV score
    mlflow.log_metric("best_cv_rmse", np.sqrt(-best_score))
    
    # Train the model with the best parameters on the full training set
    best_model = LGBMRegressor(**best_params, random_state=42)
    best_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    
    # Calculate and log test metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_rmse", rmse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)
    
    # Log the model
    mlflow.sklearn.log_model(best_model, "lgbm_best_model")
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")