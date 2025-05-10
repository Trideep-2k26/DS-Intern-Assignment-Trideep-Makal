import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib


def train_model(data/modified_energy_dataset (1).csv):
    # Load data
    df = pd.read_csv("data/modified_energy_dataset (1).csv")

    # Define features and target
    features = [
        'lighting_energy', 'zone1_temperature', 'zone2_temperature',
        'zone2_humidity', 'zone3_temperature', 'zone3_humidity',
        'zone4_temperature', 'zone4_humidity', 'zone5_temperature',
        'zone5_humidity', 'zone6_temperature', 'zone6_humidity',
        'zone7_temperature', 'zone7_humidity', 'zone8_temperature',
        'zone8_humidity', 'zone9_temperature', 'zone9_humidity',
        'outdoor_temperature', 'atmospheric_pressure', 'outdoor_humidity',
        'wind_speed', 'visibility_index', 'dew_point',
        'random_variable1', 'random_variable2', 'hour', 'day', 'month'
    ]
    target = 'equipment_energy_consumption'

    # Train model
    X = df[features]
    y = df[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model
    joblib.dump(model, 'energy_model.pkl')
    joblib.dump(features, 'model_features.pkl')

    return "Model trained and saved successfully!"


if __name__ == "__main__":
    dataset_path = "energy_data.csv"  # Default path
    print(train_model(dataset_path))