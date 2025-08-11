import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and tools
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")  # Dictionary of LabelEncoders

# Feature during training
required_columns = ['day_of_week', 'location', 'is_holiday', 'weather_condition',
                    'total_slots', 'event_nearby', 'start_time', 'end_time']

# Location-based total slot auto-fill
location_slot_map = {
    'Orion Mall, Malleshwaram': 5200,
    'Phoenix Marketcity, Whitefield': 1700,
    'Mantri Square Mall, Malleswaram': 2350,
    'Total Mall, Old Airport Road': 300,
    'Brigade Road': 85,
    'Commercial Street': 75,
    'Church Street': 500,
    'Indiranagar 100 Feet Road': 400,
    'Jayanagar 4th Block': 350,
    'Manyata Tech Park, Hebbal': 1200,
    'International Tech Park (ITPL), Whitefield': 1500,
    'JC Road Multi-Level Parking': 300,
    'KR Market Parking Complex': 370,
    'Freedom Park': 40,
    'Citywide Parking Overview': 5300
}

# Time split
def time_str_to_hour(time_str):
    hour, minute = map(int, time_str.split(":"))
    return hour + (minute / 60)

# UI Setup
st.set_page_config(page_title="🚗 Bengaluru Smart Parking Predictor", layout="wide")
st.title("🚗 Bengaluru Smart Parking Availability Predictor 🛵")


st.sidebar.header("📍 Input Features")

start_time_str = st.sidebar.selectbox("⏰ Start Time", ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00',
                                                       '12:00', '14:00', '16:00', '18:00', '20:00', '22:00'])
end_time_str = st.sidebar.selectbox("⏳ End Time", ['02:00', '04:00', '06:00', '08:00', '10:00', '12:00',
                                                   '14:00', '16:00', '18:00', '20:00', '22:00', '00:00'])
day = st.sidebar.selectbox("📅 Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
location = st.sidebar.selectbox("📍 Location", list(location_slot_map.keys()))
is_holiday = st.sidebar.selectbox("🏖️ Is it a holiday?", ['No', 'Yes'])
weather = st.sidebar.selectbox("🌦️ Weather Condition", ['Sunny', 'Rainy', 'Cloudy', 'Foggy'])
event_nearby = st.sidebar.selectbox("🎉 Event Nearby?", ['No', 'Yes'])

# Auto-fill total slots
total_slots = location_slot_map.get(location, 500)

# Convert start/end time to numeric
start_time = time_str_to_hour(start_time_str)
end_time = time_str_to_hour(end_time_str)

# Create input DataFrame
input_data = {
    'day_of_week': [day],
    'location': [location],
    'is_holiday': [1 if is_holiday == 'Yes' else 0],
    'weather_condition': [weather],
    'total_slots': [total_slots],
    'event_nearby': [1 if event_nearby == 'Yes' else 0],
    'start_time': [start_time],
    'end_time': [end_time],
}
input_df = pd.DataFrame(input_data)

# Apply Label Encoding
for col in input_df.select_dtypes(include='object').columns:
    if col in encoder:
        input_df[col] = encoder[col].transform(input_df[col])
    else:
        st.error(f"⚠️ Missing encoder for column: {col}")
        st.stop()

# Ensure correct column order AFTER encoding
input_df = input_df[required_columns]


# Scale input
scaled_input = scaler.transform(input_df)

# Predict on button click
if st.button("🔮 Predict Parking Availability"):
    rf_pred = rf_model.predict(scaled_input)
    xgb_pred = xgb_model.predict(scaled_input)

    # Round and clip predictions to avoid negatives
    rf_car_pred = max(0, int(np.round(rf_pred[0][0])))
    rf_bike_pred = max(0, int(np.round(rf_pred[0][1])))
    xgb_car_pred = max(0, int(np.round(xgb_pred[0][0])))
    xgb_bike_pred = max(0, int(np.round(xgb_pred[0][1])))

    # Show results
    col1, col2 = st.columns(2)
    with col1:
        st.success("🚗 EcoPark RF Prediction")
        st.metric("4-Wheeler Slots Available", rf_car_pred)
        st.metric("2-Wheeler Slots Available", rf_bike_pred)
    with col2:
        st.info("🛵 SmartXGB Park Prediction")
        st.metric("4-Wheeler Slots Available", xgb_car_pred)
        st.metric("2-Wheeler Slots Available", xgb_bike_pred)
