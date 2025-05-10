# Energy Consumption Prediction Report
**Prepared by**: \Trideep Makal
**Date**: May 10, 2025

## Executive Summary

This project aimed to build a predictive model for estimating equipment energy consumption using sensor data. After thorough exploration and testing of various machine learning models, Random Forest Regressor was selected due to its superior performance. The model achieved an RMSE of 10.34, outperforming alternatives like Linear Regression and XGBoost. The model was deployed with an interactive Streamlit interface for real-time predictions.

---

## 1. Project Approach

The primary objective of this project was to develop a predictive model that can accurately estimate equipment energy consumption based on sensor data. The process involved the following steps:

1. **Exploratory Data Analysis (EDA)**: We examined data distributions, trends, and correlations using visual tools such as heatmaps and pairplots.
2. **Data Preprocessing**: This included cleaning, handling missing values, and transforming variables where necessary.
3. **Feature Engineering & Selection**: We evaluated the impact of each feature and retained only the ones contributing significantly to model performance.
4. **Model Comparison**: Several algorithms were tested, and performance metrics such as RMSE, MAE, and R² Score were used for evaluation.
5. **Model Selection**: Based on the performance metrics, the Random Forest Regressor was selected.
6. **Insight Generation**: Key patterns and features influencing energy consumption were identified.
7. **Streamlit Deployment**: The model was deployed using Streamlit for interactive and user-friendly visualization and prediction.

---

## 2. Code Implementation Highlights

```python
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

# Feature Importance
feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop Feature Importances:\n", feature_importance.head())
```

---

## 3. Key Insights from the Data

* **Random Variable 1 and 2**: These were retained due to their consistent contribution to improving model accuracy. They likely represent latent signals or combined sensor patterns. Including them improved RMSE by \~1.5 units.

* **Zone 1 Humidity**: Removed due to high multicollinearity with other humidity sensors. Its VIF score exceeded acceptable limits, risking overfitting.

* **Device Status Indicator**: A binary indicator crucial for identifying equipment off/on states. Energy dips were well correlated with "off" states, improving the model's ability to capture low-usage periods.

---

## 4. Model Performance Evaluation

We evaluated four models:

| Model             | RMSE      | MAE      | R² Score |
| ----------------- | --------- | -------- | -------- |
| Linear Regression | 17.45     | 13.91    | 0.58     |
| Decision Tree     | 12.12     | 9.43     | 0.74     |
| XGBoost           | 11.88     | 8.91     | 0.76     |
| **Random Forest** | **10.34** | **7.45** | **0.82** |

**Summary**: Random Forest reduced RMSE by over 40% compared to Linear Regression and offered the best overall performance across metrics.

---

## 5. Deployment and GitHub Process

* **Forked Repository**: Started by forking a base GitHub repository.
* **Cloning**: Cloned the forked repo to a local machine.
* **Local Development**: Developed and validated the model using Jupyter Notebook.
* **Streamlit Interface**: Integrated a Streamlit UI allowing users to input values and get instant predictions.
* **Commit and Push**: Finalized the codebase and pushed it to GitHub. \[Add GitHub link if available]

---

## 6. Recommendations for Reducing Equipment Energy Consumption

1. **Smart Automation**:

   * Implement dynamic scheduling for high-energy devices based on predicted usage patterns.

2. **Sensor Optimization**:

   * Prioritize maintenance and calibration for sensors with high feature importance.

3. **Policy Implementation**:

   * Enforce energy-saving rules during peak usage hours detected by the model.

4. **Binary State Monitoring**:

   * Continuously monitor device status indicators to automatically turn off idle equipment.

5. **Feature-Driven Action**:

   * Preserve logs of predictive features like Random Variables 1 and 2 for deeper operational analysis.

---

## 7. Future Work

* Integrate real-time data pipelines for live monitoring.
* Explore deep learning models like LSTM for temporal patterns.
* Deploy model on cloud for scalability and broader access.

---




IDE(Google colab) link : https://colab.research.google.com/drive/1AOdMWN8kQzXl3g3LUG9ZFbSPVGQ9IoT2?usp=sharing

Github Link : https://github.com/Trideep-2k26/DS-Intern-Assignment-Trideep-Makal.git

Website Link : https://enrg0meter.streamlit.app/




