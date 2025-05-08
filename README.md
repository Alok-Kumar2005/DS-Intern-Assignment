# Smart Factory Energy Prediction Challenge



```
conda create -n DSInter python=3.10 -y
conda activate DSInter
pip install -r requirements.txt
```



## Approach to the Problem
### Data Ingestion:
   - Loaded raw data from CSV files
   - Performed basic timestamp conversion and extracted time features (hour, minute)
   - Focused on key numeric columns like equipment energy consumption, lighting energy, and temperature/humidity zones
   - Split data into train and test sets for model development


## Data Preprocessing:
   - Identified and removed low-correlation features with threshold less than 0.01
   - Handled outliers using a clamping technique based on IQR (Interquartile Range)
   - Consolidated random variables by taking their average
   - Dropped redundant columns to focus on the most relevant features


## Feature Engineering:
   - Applied KNN imputation to handle missing values in both train and test datasets
   - Used standardization to normalize feature scales
   - Extracted the target variable (equipment_energy_consumption) from the datasets
   - Saved the processed features for model training


## Model Building:
   - Trained a LightGBM regression model, a gradient boosting framework optimized for efficiency
   - Utilized parameters specified in the configuration file
   - Evaluated initial model performance on training data using RMSE and R² metrics


## Model Evaluation:
   - Assessed model performance on test data
   - Calculated key metrics: mean squared error, mean absolute error, and R² score
   - Saved results to a metrics file for tracking and comparison

# Key Insights from the Data
   - there are lots of column which are not helping in predicting the output
   - 'random_variable1' and 'random_variable2' is similar
   - outlier present in the data
   - almost all columns are normalized

# Model perfomace evaluation
   - using lazyregressor to predict the best model
   - LGBMClassifier and some other are also performing good on the dataset
   - using MLFlow tool to log the value 
   - with the help of RandomSearchCV got the best parameter's

# Recommendations for reducing equipment energy consumption
### Time-Based Equipment Scheduling:
   - Use the hourly patterns identified to create optimized equipment schedules
   - Program equipment to operate in low-power modes during typical low-activity hours

### Continuous Monitoring and Optimization:
   - Deploy the model in production to continuously monitor energy consumption
   - Set up alerts for unexpected spikes in usage
   - Regularly retrain the model with new data to maintain accuracy

## Two columns `random_varaible1` and `random_varaible1` are similar to each other so
- took the average of the both 
- add a new column named `random_variable` and then remove the earlier one



## check the model performace and hyperparamter tuning at MLflow
`https://dagshub.com/ay747283/DS-Intern-Assignment.mlflow`


## to check the output of the code run
- fork repositoy and then run commad 
```
dvc repro
python app.py
```