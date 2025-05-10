Energy Consumption Prediction Report

 Prepared by:Trideep Makal
 Date: May 10, 2025
1. Project Approach
The primary objective of this project was to develop a predictive model that can accurately estimate equipment energy consumption based on sensor data. The process involved the following steps:
Exploratory Data Analysis (EDA): We examined data distributions, trends, and correlations using visual tools such as heatmaps and pairplots.


Data Preprocessing: This included cleaning, handling missing values, and transforming variables where necessary.


Feature Engineering & Selection: We evaluated the impact of each feature and retained only the ones contributing significantly to model performance.


Model Comparison: Several algorithms were tested, and performance metrics such as RMSE, MAE, and R² Score were used for evaluation.


Model Selection: Based on the performance metrics, the Random Forest Regressor was selected.


Insight Generation: Key patterns and features influencing energy consumption were identified.


Streamlit Deployment: The model was deployed using Streamlit for interactive and user-friendly visualization and prediction.



2. Code Implementation Highlights
# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv("energy_data.csv")

# Drop Zone 1 Humidity due to multicollinearity
df = df.drop(columns=['Zone1_Humidity'])

# Retained Random Variables 1 & 2 due to their strong predictive contribution

# Train-test Split
X = df.drop(columns=['Energy_Consumption'])
y = df['Energy_Consumption']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluation
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("MAE:", mae)
print("R²:", r2)


3. Key Insights from the Data
Random Variable 1 and 2: These were retained due to their consistent contribution to improving model accuracy. They may represent latent patterns or complex equipment behavior. Their values varied but consistently improved metrics: keeping them reduced RMSE by ~1.5 units.


Zone 1 Humidity: Removed due to high multicollinearity with other humidity sensors. Its VIF score exceeded acceptable limits, risking overfitting.


False Values Column: Binary indicator crucial for identifying equipment off/on states. Energy dips were well correlated with False, improving the model's ability to capture low-usage periods.



4. Model Performance Evaluation
We evaluated four models:
Model
RMSE
MAE
R² Score
Linear Regression
17.45
13.91
0.58
Decision Tree
12.12
9.43
0.74
XGBoost
11.88
8.91
0.76
Random Forest
10.34
7.45
0.82

Why Random Forest?
Outperformed other models on all evaluation metrics.


Robust against overfitting due to ensemble nature.


Effectively handles missing and noisy data.


Captures non-linear dependencies better than linear models.



5. Deployment and GitHub Process
Forked Repository: Initially forked a base GitHub repository from another account.


Cloning: Cloned the forked repo to local machine.


Local Development: Ran and tested the code locally using Jupyter Notebook.


Streamlit Interface: Integrated a Streamlit UI for better navigation and frontend interaction. Users can input values and receive predictions in real-time.


Commit and Push: After verifying performance, the final codebase and Streamlit app were committed and pushed to the forked GitHub repository.



6. Recommendations for Reducing Equipment Energy Consumption
Smart Automation:


Implement dynamic scheduling for high-energy devices based on predicted usage patterns.


Sensor Optimization:


Focus maintenance and calibration on sensors with high importance scores (e.g., device state sensors, motion sensors).


Policy Implementation:


Enforce energy-saving policies during peak usage hours detected by the model.


Binary State Monitoring:


Actively monitor and respond to device states represented in the 'False values' column to switch off unused equipment in real time.


Feature-Driven Action:


Retain usage logs of features like Random Variables 1 and 2 until deeper root-cause analysis is possible.




Github Link : https://github.com/Trideep-2k26/DS-Intern-Assignment-Trideep-Makal.git
Website Link : https://enrg0meter.streamlit.app/




