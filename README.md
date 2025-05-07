# Smart Factory Energy Prediction Challenge



```
conda create -n DSInter python=3.10 -y
conda activate DSInter
pip install -r requirements.txt
```




## Problem Overview

You've been hired as a data scientist for SmartManufacture Inc., a leading industrial automation company. The company has deployed an extensive sensor network throughout one of their client's manufacturing facilities to monitor environmental conditions and energy usage.

The client is concerned about the increasing energy costs associated with their manufacturing equipment. They want to implement a predictive system that can forecast equipment energy consumption based on various environmental factors and sensor readings from different zones of the factory.

## Your Task

Your assignment is to develop a machine learning model that can accurately predict the energy consumption of industrial equipment (`equipment_energy_consumption`) based on the data collected from the factory's sensor network. This will help the facility managers optimize their operations for energy efficiency and cost reduction.

### Specific Goals:

1. Analyze the provided sensor data to identify patterns and relationships between environmental factors and equipment energy consumption
2. Build a robust regression model to predict equipment energy consumption
3. Evaluate the model's performance using appropriate metrics
4. Provide actionable insights and recommendations for reducing energy consumption

## Repository Structure

This repository is organized as follows:

```
.
├── data/               # Contains the training and test datasets
│   ├── data.csv        # dataset
├── docs/               # Documentation files
│   └── data_description.md  # Detailed description of all features
└── README.md           # This file
```

## Dataset Description

The data comes from a manufacturing facility equipped with multiple sensors that collect environmental measurements. Each record contains:

- Timestamp of the measurement
- Energy consumption readings for equipment and lighting
- Temperature and humidity readings from 9 different zones in the facility
- Outdoor weather conditions (temperature, humidity, pressure, etc.)
- Additional measurements and calculated variables

### Notes on Feature Selection and Random Variables

The dataset includes two variables named `random_variable1` and `random_variable2`. Part of your task is to determine, through proper data analysis and feature selection techniques, whether these variables should be included in your model or not. This mimics real-world scenarios where not all available data is necessarily useful for prediction.

Your approach to handling these variables should be clearly documented and justified in your analysis. This will be an important part of evaluating your feature selection methodology.

Note that your final solution will also be evaluated on a separate holdout dataset that we maintain privately, which serves as an additional check on your model's generalization capability.

For a detailed description of all features, please refer to the [data description document](docs/data_description.md).

## Deliverables

Your submission should include:

1. **A well-documented Jupyter notebook** containing:
   - Exploratory data analysis (EDA)
   - Data preprocessing steps
   - Feature engineering and selection
   - Model development and training
   - Model evaluation and testing
   - Key findings and insights

2. **Python script(s)/notebook(s)** with your final model implementation

3. **A brief report (PDF or Markdown format)** summarizing:
   - Your approach to the problem
   - Key insights from the data
   - Model performance evaluation
   - Recommendations for reducing equipment energy consumption

## Evaluation Criteria

Your solution will be evaluated based on:

1. **Code Quality and Structure (25%)**
   - Clean, well-organized, and properly documented code
   - Appropriate use of functions and classes
   - Effective use of Git with meaningful commit messages
   - Code readability and adherence to Python conventions

2. **Data Analysis and Preprocessing (25%)**
   - Thoroughness of exploratory data analysis
   - Handling of missing values, outliers, and data transformations
   - Feature engineering creativity and effectiveness
   - Proper data splitting methodology

3. **Model Development (25%)**
   - Selection and justification of algorithms
   - Hyperparameter tuning approach
   - Implementation of cross-validation
   - Model interpretability considerations

4. **Results and Insights (25%)**
   - Model performance metrics (RMSE, MAE, R²) on both the test dataset and our private holdout dataset
   - Quality of visualizations and explanations
   - Practical insights and recommendations
   - Critical evaluation of model limitations

## Submission Instructions

1. Fork this repository to your own GitHub account, naming it `DS-Intern-Assignment-[YourName]` (replace `[YourName]` with your actual name)
2. Clone your forked repository to your local machine
3. Make regular, meaningful commits as you develop your solution
4. Push your changes to your forked repository
5. Once complete, submit the URL of your forked repository via replying to the mail.

Your commit history will be reviewed as part of the evaluation, so make sure to commit regularly and include meaningful commit messages that reflect your development process.

## Time Commitment

This assignment is designed to be completed in approximately 4-6 hours.
Deadline is 48 hours/2 days from when you receive the assignment.

Good luck!
