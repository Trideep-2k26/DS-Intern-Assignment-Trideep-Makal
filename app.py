import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# Set page configuration
st.set_page_config(
    page_title="SmartManufacture Energy Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; 
        color: #1E88E5; 
        text-align: center; 
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem; 
        color: #0D47A1; 
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .info-box, .cost-box, .prediction-box {
        background-color: #ffffff; /* White background */
        color: #000000; /* Black text */
        padding: 20px; 
        border-radius: 10px; 
        margin-bottom: 20px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .cost-box h3, .cost-box h4, 
    .prediction-box h3, .prediction-box h2 {
        color: #000000; /* Black text for headings */
    }
    .prediction-box h1, .prediction-box h2 {
        color: #2E7D32 !important; /* Green for predicted values */
    }
</style>
""", unsafe_allow_html=True)


def load_model():
    try:
        model = joblib.load('energy_model.pkl')
        features = joblib.load('model_features.pkl')
        return model, features, True
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, False


def main():
    st.markdown("<div class='main-header'>⚡ SmartManufacture Energy Predictor</div>", unsafe_allow_html=True)



    # Check if model exists
    model, features, model_loaded = load_model()
    if not model_loaded:
        st.warning("Please train the model first using the backend script.")
        return

    forecast_method = st.radio(
        "Choose forecasting method:",
        ["Upload a CSV file", "Enter parameters manually"],
        horizontal=True
    )

    if forecast_method == "Upload a CSV file":
        forecast_file = st.file_uploader("Upload data for forecasting (CSV)", type=["csv"])

        if forecast_file is not None:
            forecast_df = pd.read_csv(forecast_file)
            st.success("✅ Forecast data uploaded successfully!")

            # Check features
            missing_features = [f for f in features if f not in forecast_df.columns]
            if missing_features:
                st.error(f"Missing features: {', '.join(missing_features)}")
                return

            # Energy rate input
            energy_rate = st.number_input(
                "Energy cost per unit ($/kWh)",
                min_value=0.0, value=0.15, step=0.01, format="%.4f"
            )

            if st.button("Generate Forecasts"):
                predictions = model.predict(forecast_df[features])
                energy_costs = predictions * energy_rate

                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class='cost-box'>
                        <h4>Maximum Cost</h4>
                        <h3>${np.max(energy_costs):,.2f}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class='cost-box'>
                        <h4>Average Cost</h4>
                        <h3>${np.mean(energy_costs):,.2f}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class='cost-box'>
                        <h4>Total Cost</h4>
                        <h3>${np.sum(energy_costs):,.2f}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                # Visualization
                fig = px.line(
                    x=forecast_df.index,
                    y=predictions,
                    labels={'x': 'Time', 'y': 'Energy Consumption (kWh)'},
                    title="Predicted Equipment Energy Consumption"
                )
                st.plotly_chart(fig, use_container_width=True)

    else:  # Manual input
        st.markdown("<div class='sub-header'>Enter Parameters</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        input_data = {}

        with col1:
            input_data['month'] = st.number_input("Month", 1, 12, 1)
            input_data['day'] = st.number_input("Day", 1, 31, 1)
            input_data['hour'] = st.number_input("Hour", 0, 23, 12)
            input_data['lighting_energy'] = st.number_input("Lighting Energy", 0.0, 1000.0, 50.0)
            input_data['zone1_temperature'] = st.number_input("Zone 1 Temp", -20.0, 50.0, 25.0)
            input_data['zone2_temperature'] = st.number_input("Zone 2 Temp", -20.0, 50.0, 25.0)
            input_data['zone2_humidity'] = st.number_input("Zone 2 Humidity", 0.0, 100.0, 50.0)
            input_data['zone3_temperature'] = st.number_input("Zone 3 Temp", -20.0, 50.0, 25.0)
            input_data['zone3_humidity'] = st.number_input("Zone 3 Humidity", 0.0, 100.0, 50.0)


        with col2:



            input_data['zone4_temperature'] = st.number_input("Zone 4 Temp", -20.0, 50.0, 25.0)
            input_data['zone4_humidity'] = st.number_input("Zone 4 Humidity", 0.0, 100.0, 50.0)
            input_data['zone5_temperature'] = st.number_input("Zone 5 Temp", -20.0, 50.0, 25.0)
            input_data['zone5_humidity'] = st.number_input("Zone 5 Humidity", 0.0, 100.0, 50.0)
            input_data['zone6_temperature'] = st.number_input("Zone 6 Temp", -20.0, 50.0, 25.0)
            input_data['zone6_humidity'] = st.number_input("Zone 6 Humidity", 0.0, 100.0, 50.0)
            input_data['zone7_temperature'] = st.number_input("Zone 7 Temp", -20.0, 50.0, 25.0)
            input_data['zone7_humidity'] = st.number_input("Zone 7 Humidity", 0.0, 100.0, 50.0)
            input_data['zone8_temperature'] = st.number_input("Zone 8 Temp", -20.0, 50.0, 25.0)
            input_data['zone8_humidity'] = st.number_input("Zone 8 Humidity", 0.0, 100.0, 50.0)




        with col3:
            input_data['zone9_temperature'] = st.number_input("Zone 9 Temp", -20.0, 50.0, 25.0)
            input_data['zone9_humidity'] = st.number_input("Zone 9 Humidity", 0.0, 100.0, 50.0)





            input_data['outdoor_temperature'] = st.number_input("Outdoor Temp", -20.0, 50.0, 25.0)
            input_data['atmospheric_pressure'] = st.number_input("Pressure", 800.0, 1200.0, 1013.0)
            input_data['outdoor_humidity'] = st.number_input("Outdoor Humidity", 0.0, 100.0, 50.0)

            input_data['wind_speed'] = st.number_input("Wind Speed", 0.0, 100.0, 10.0)
            input_data['visibility_index'] = st.number_input("Visibility", 0.0, 10.0, 5.0)
            input_data['dew_point'] = st.number_input("Dew Point", -20.0, 30.0, 10.0)


        # Energy rate input
        energy_rate = st.number_input(
            "Energy cost per unit ($/kWh)",
            min_value=0.0, value=0.15, step=0.01, format="%.4f"
        )

        if st.button("Predict Energy Consumption"):
            # Create input dataframe with all features (fill others with median values)
            full_input = {f: 0 for f in features}  # Initialize with zeros
            full_input.update(input_data)  # Update with user-provided values

            prediction = model.predict(pd.DataFrame([full_input]))[0]
            energy_cost = prediction * energy_rate

            st.markdown(f"""
            <div class='prediction-box'>
                <h3>Predicted Equipment Energy Consumption</h3>
                <h1 style="color: #2E7D32;">{prediction:.2f} kWh</h1>
                <h3>Estimated Cost</h3>
                <h2 style="color: #2E7D32;">${energy_cost:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()