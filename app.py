import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import folium
from streamlit_folium import folium_static
from datetime import datetime
import math
warnings.filterwarnings('ignore')

st.set_page_config(page_title="NYC Taxi Fare Predictor", layout="wide")

st.title("NYC Taxi Fare Prediction System")

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
    
    bearing = math.atan2(x, y)
    return bearing

def calculate_airport_distances(lat, lon):
    jfk = (40.6413, -73.7781)
    ewr = (40.6895, -74.1745)
    lga = (40.7769, -73.8740)
    sol = (40.6892, -74.0445)
    nyc_center = (40.7580, -73.9855)
    
    jfk_dist = calculate_distance(lat, lon, jfk[0], jfk[1])
    ewr_dist = calculate_distance(lat, lon, ewr[0], ewr[1])
    lga_dist = calculate_distance(lat, lon, lga[0], lga[1])
    sol_dist = calculate_distance(lat, lon, sol[0], sol[1])
    nyc_dist = calculate_distance(lat, lon, nyc_center[0], nyc_center[1])
    
    return jfk_dist, ewr_dist, lga_dist, sol_dist, nyc_dist

@st.cache_resource
def load_model():
    try:
        model = joblib.load('D://uper_predication//uber-predication//best_model.pkl')
        return model
    except:
        st.error("Model file not found. Please make sure 'best_model.pkl' is in the same directory.")
        return None

if 'pickup_coords' not in st.session_state:
    st.session_state.pickup_coords = [40.7580, -73.9855]
if 'dropoff_coords' not in st.session_state:
    st.session_state.dropoff_coords = [40.7489, -73.9680]
if 'trip_history' not in st.session_state:
    st.session_state.trip_history = []

model = load_model()

tabs = st.tabs(["Single Prediction", "Bulk Prediction", "Trip History", "Statistics"])

with tabs[0]:
    if model is not None:
        st.success("Model loaded successfully")
        
        col_map, col_form = st.columns([1.5, 1])
        
        with col_map:
            st.subheader("Select Pickup & Dropoff Locations")
            
            map_center = [40.7580, -73.9855]
            m = folium.Map(location=map_center, zoom_start=12)
            
            folium.Marker(
                st.session_state.pickup_coords,
                popup="Pickup Location",
                icon=folium.Icon(color='green', icon='play'),
                draggable=False
            ).add_to(m)
            
            folium.Marker(
                st.session_state.dropoff_coords,
                popup="Dropoff Location",
                icon=folium.Icon(color='red', icon='stop'),
                draggable=False
            ).add_to(m)
            
            folium.PolyLine(
                [st.session_state.pickup_coords, st.session_state.dropoff_coords],
                color='blue',
                weight=3,
                opacity=0.7
            ).add_to(m)
            
            folium_static(m, width=700, height=500)
            
            st.info("Use the form below to enter exact coordinates or use popular locations")
            
            popular_locations = {
                "Times Square": [40.7580, -73.9855],
                "JFK Airport": [40.6413, -73.7781],
                "LaGuardia Airport": [40.7769, -73.8740],
                "Newark Airport": [40.6895, -74.1745],
                "Statue of Liberty": [40.6892, -74.0445],
                "Central Park": [40.7829, -73.9654],
                "Brooklyn Bridge": [40.7061, -73.9969],
                "Empire State Building": [40.7484, -73.9857]
            }
            
            col_pick, col_drop = st.columns(2)
            
            with col_pick:
                pickup_location = st.selectbox("Quick Pickup Location", ["Custom"] + list(popular_locations.keys()), key='pickup_select')
                if pickup_location != "Custom":
                    st.session_state.pickup_coords = popular_locations[pickup_location]
                
                pickup_lat = st.number_input("Pickup Latitude", value=st.session_state.pickup_coords[0], format="%.6f", key='pick_lat')
                pickup_lon = st.number_input("Pickup Longitude", value=st.session_state.pickup_coords[1], format="%.6f", key='pick_lon')
                
                if st.button("Update Pickup", key='update_pickup'):
                    st.session_state.pickup_coords = [pickup_lat, pickup_lon]
                    st.rerun()
            
            with col_drop:
                dropoff_location = st.selectbox("Quick Dropoff Location", ["Custom"] + list(popular_locations.keys()), key='dropoff_select')
                if dropoff_location != "Custom":
                    st.session_state.dropoff_coords = popular_locations[dropoff_location]
                
                dropoff_lat = st.number_input("Dropoff Latitude", value=st.session_state.dropoff_coords[0], format="%.6f", key='drop_lat')
                dropoff_lon = st.number_input("Dropoff Longitude", value=st.session_state.dropoff_coords[1], format="%.6f", key='drop_lon')
                
                if st.button("Update Dropoff", key='update_dropoff'):
                    st.session_state.dropoff_coords = [dropoff_lat, dropoff_lon]
                    st.rerun()
        
        with col_form:
            st.subheader("Trip Details")
            
            pickup_lat_rad = math.radians(st.session_state.pickup_coords[0])
            pickup_lon_rad = math.radians(st.session_state.pickup_coords[1])
            dropoff_lat_rad = math.radians(st.session_state.dropoff_coords[0])
            dropoff_lon_rad = math.radians(st.session_state.dropoff_coords[1])
            
            calculated_distance = calculate_distance(
                st.session_state.pickup_coords[0],
                st.session_state.pickup_coords[1],
                st.session_state.dropoff_coords[0],
                st.session_state.dropoff_coords[1]
            )
            
            calculated_bearing = calculate_bearing(
                st.session_state.pickup_coords[0],
                st.session_state.pickup_coords[1],
                st.session_state.dropoff_coords[0],
                st.session_state.dropoff_coords[1]
            )
            
            jfk_dist, ewr_dist, lga_dist, sol_dist, nyc_dist = calculate_airport_distances(
                st.session_state.pickup_coords[0],
                st.session_state.pickup_coords[1]
            )
            
            st.metric("Calculated Distance", f"{calculated_distance:.2f} km")
            st.metric("Bearing", f"{calculated_bearing:.4f} rad")
            
            car_condition = st.selectbox("Car Condition", ['Bad', 'Excellent', 'Good', 'Very Good'])
            weather = st.selectbox("Weather", ['cloudy', 'rainy', 'stormy', 'sunny', 'windy'])
            traffic = st.selectbox("Traffic Condition", ['Congested Traffic', 'Dense Traffic', 'Flow Traffic'])
            
            passenger_count = st.number_input("Passenger Count", min_value=0, max_value=6, value=1)
            
            use_current_time = st.checkbox("Use Current Date & Time", value=True)
            
            if use_current_time:
                now = datetime.now()
                hour = now.hour
                day = now.day
                month = now.month
                weekday = now.weekday()
                year = now.year
                
                st.info(f"Current: {now.strftime('%Y-%m-%d %H:%M')}")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    hour = st.slider("Hour", 0, 23, 12)
                    day = st.slider("Day", 1, 31, 15)
                    month = st.slider("Month", 1, 12, 6)
                with col2:
                    weekday = st.slider("Weekday (0=Mon)", 0, 6, 3)
                    year = st.number_input("Year", min_value=2009, max_value=2025, value=2015)
            
            if st.button("Predict Fare", type="primary", use_container_width=True):
                car_condition_map = {'Bad': 0.0, 'Excellent': 1.0, 'Good': 2.0, 'Very Good': 3.0}
                traffic_map = {'Congested Traffic': 0.0, 'Dense Traffic': 1.0, 'Flow Traffic': 2.0}
                
                day_or_night_pm = 1.0 if hour >= 12 else 0.0
                is_alone_true = 1.0 if passenger_count == 1 else 0.0
                
                input_data = {
                    'Car Condition': car_condition_map[car_condition],
                    'Traffic Condition': traffic_map[traffic],
                    'pickup_longitude': pickup_lon_rad,
                    'pickup_latitude': pickup_lat_rad,
                    'dropoff_longitude': dropoff_lon_rad,
                    'dropoff_latitude': dropoff_lat_rad,
                    'passenger_count': passenger_count,
                    'hour': hour,
                    'day': day,
                    'month': month,
                    'weekday': weekday,
                    'year': year,
                    'jfk_dist': jfk_dist,
                    'ewr_dist': ewr_dist,
                    'lga_dist': lga_dist,
                    'sol_dist': sol_dist,
                    'nyc_dist': nyc_dist,
                    'distance': calculated_distance,
                    'bearing': calculated_bearing,
                    'Weather_rainy': 1.0 if weather == 'rainy' else 0.0,
                    'Weather_stormy': 1.0 if weather == 'stormy' else 0.0,
                    'Weather_sunny': 1.0 if weather == 'sunny' else 0.0,
                    'Weather_windy': 1.0 if weather == 'windy' else 0.0,
                    'day_or_night_pm': day_or_night_pm,
                    'is_alone_true': is_alone_true
                }
                
                input_df = pd.DataFrame([input_data])
                
                prediction_scaled = model.predict(input_df)
                
                fare_min = 2.5
                fare_max = 52.0
                predicted_fare = prediction_scaled[0] * (fare_max - fare_min) + fare_min
                
                st.success(f"Predicted Fare: ${predicted_fare:.2f}")
                
                trip_record = {
                    'Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'From': f"({st.session_state.pickup_coords[0]:.4f}, {st.session_state.pickup_coords[1]:.4f})",
                    'To': f"({st.session_state.dropoff_coords[0]:.4f}, {st.session_state.dropoff_coords[1]:.4f})",
                    'Distance': f"{calculated_distance:.2f} km",
                    'Fare': f"${predicted_fare:.2f}",
                    'Weather': weather,
                    'Traffic': traffic
                }
                st.session_state.trip_history.insert(0, trip_record)
                
                with st.expander("Fare Breakdown"):
                    st.write("**Factors affecting the fare:**")
                    st.write(f"- Base Distance: {calculated_distance:.2f} km")
                    st.write(f"- Time of Day: {'PM' if hour >= 12 else 'AM'}")
                    st.write(f"- Weather Condition: {weather}")
                    st.write(f"- Traffic Condition: {traffic}")
                    st.write(f"- Passengers: {passenger_count}")
                    st.write(f"- Car Condition: {car_condition}")
    else:
        st.error("Please add 'best_model.pkl' file to the directory and restart the app.")

with tabs[1]:
    st.subheader("Bulk Prediction from CSV")
    
    st.write("Upload a CSV file with the following columns:")
    st.code("pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, Car Condition, Weather, Traffic Condition, passenger_count, hour, day, month, weekday, year")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())
        
        if st.button("Predict All", type="primary"):
            if model is not None:
                predictions = []
                
                progress_bar = st.progress(0)
                for idx, row in df.iterrows():
                    pickup_lat_rad = math.radians(row['pickup_latitude'])
                    pickup_lon_rad = math.radians(row['pickup_longitude'])
                    dropoff_lat_rad = math.radians(row['dropoff_latitude'])
                    dropoff_lon_rad = math.radians(row['dropoff_longitude'])
                    
                    calculated_distance = calculate_distance(
                        row['pickup_latitude'], row['pickup_longitude'],
                        row['dropoff_latitude'], row['dropoff_longitude']
                    )
                    
                    calculated_bearing = calculate_bearing(
                        row['pickup_latitude'], row['pickup_longitude'],
                        row['dropoff_latitude'], row['dropoff_longitude']
                    )
                    
                    jfk_dist, ewr_dist, lga_dist, sol_dist, nyc_dist = calculate_airport_distances(
                        row['pickup_latitude'], row['pickup_longitude']
                    )
                    
                    car_condition_map = {'Bad': 0.0, 'Excellent': 1.0, 'Good': 2.0, 'Very Good': 3.0}
                    traffic_map = {'Congested Traffic': 0.0, 'Dense Traffic': 1.0, 'Flow Traffic': 2.0}
                    
                    input_data = {
                        'Car Condition': car_condition_map.get(row['Car Condition'], 2.0),
                        'Traffic Condition': traffic_map.get(row['Traffic Condition'], 1.0),
                        'pickup_longitude': pickup_lon_rad,
                        'pickup_latitude': pickup_lat_rad,
                        'dropoff_longitude': dropoff_lon_rad,
                        'dropoff_latitude': dropoff_lat_rad,
                        'passenger_count': row['passenger_count'],
                        'hour': row['hour'],
                        'day': row['day'],
                        'month': row['month'],
                        'weekday': row['weekday'],
                        'year': row['year'],
                        'jfk_dist': jfk_dist,
                        'ewr_dist': ewr_dist,
                        'lga_dist': lga_dist,
                        'sol_dist': sol_dist,
                        'nyc_dist': nyc_dist,
                        'distance': calculated_distance,
                        'bearing': calculated_bearing,
                        'Weather_rainy': 1.0 if row['Weather'] == 'rainy' else 0.0,
                        'Weather_stormy': 1.0 if row['Weather'] == 'stormy' else 0.0,
                        'Weather_sunny': 1.0 if row['Weather'] == 'sunny' else 0.0,
                        'Weather_windy': 1.0 if row['Weather'] == 'windy' else 0.0,
                        'day_or_night_pm': 1.0 if row['hour'] >= 12 else 0.0,
                        'is_alone_true': 1.0 if row['passenger_count'] == 1 else 0.0
                    }
                    
                    input_df = pd.DataFrame([input_data])
                    prediction_scaled = model.predict(input_df)
                    
                    fare_min = 2.5
                    fare_max = 52.0
                    predicted_fare = prediction_scaled[0] * (fare_max - fare_min) + fare_min
                    
                    predictions.append(predicted_fare)
                    progress_bar.progress((idx + 1) / len(df))
                
                df['Predicted_Fare'] = predictions
                
                st.success(f"Predicted {len(predictions)} trips successfully!")
                st.dataframe(df)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv',
                )
                
                st.subheader("Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Fare", f"${df['Predicted_Fare'].mean():.2f}")
                with col2:
                    st.metric("Max Fare", f"${df['Predicted_Fare'].max():.2f}")
                with col3:
                    st.metric("Min Fare", f"${df['Predicted_Fare'].min():.2f}")

with tabs[2]:
    st.subheader("Trip History")
    
    if len(st.session_state.trip_history) > 0:
        history_df = pd.DataFrame(st.session_state.trip_history)
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("Clear History"):
            st.session_state.trip_history = []
            st.rerun()
        
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download History",
            data=csv,
            file_name='trip_history.csv',
            mime='text/csv',
        )
    else:
        st.info("No trips recorded yet. Make some predictions to see them here!")

with tabs[3]:
    st.subheader("Price Comparison Tool")
    
    st.write("Compare fares across different conditions:")
    
    compare_distance = st.slider("Distance (km)", 0.0, 20.0, 5.0, 0.5)
    
    if model is not None and st.button("Generate Comparison"):
        comparison_data = []
        
        for weather in ['sunny', 'cloudy', 'rainy', 'stormy', 'windy']:
            for traffic in ['Flow Traffic', 'Dense Traffic', 'Congested Traffic']:
                for hour in [6, 12, 18, 22]:
                    input_data = {
                        'Car Condition': 2.0,
                        'Traffic Condition': {'Congested Traffic': 0.0, 'Dense Traffic': 1.0, 'Flow Traffic': 2.0}[traffic],
                        'pickup_longitude': -1.291230,
                        'pickup_latitude': 0.711269,
                        'dropoff_longitude': -1.291218,
                        'dropoff_latitude': 0.711275,
                        'passenger_count': 1,
                        'hour': hour,
                        'day': 15,
                        'month': 6,
                        'weekday': 3,
                        'year': 2015,
                        'jfk_dist': 42.5,
                        'ewr_dist': 34.8,
                        'lga_dist': 19.6,
                        'sol_dist': 18.3,
                        'nyc_dist': 10.5,
                        'distance': compare_distance,
                        'bearing': 0.0,
                        'Weather_rainy': 1.0 if weather == 'rainy' else 0.0,
                        'Weather_stormy': 1.0 if weather == 'stormy' else 0.0,
                        'Weather_sunny': 1.0 if weather == 'sunny' else 0.0,
                        'Weather_windy': 1.0 if weather == 'windy' else 0.0,
                        'day_or_night_pm': 1.0 if hour >= 12 else 0.0,
                        'is_alone_true': 1.0
                    }
                    
                    input_df = pd.DataFrame([input_data])
                    prediction_scaled = model.predict(input_df)
                    predicted_fare = prediction_scaled[0] * (52.0 - 2.5) + 2.5
                    
                    comparison_data.append({
                        'Weather': weather,
                        'Traffic': traffic,
                        'Hour': hour,
                        'Fare': predicted_fare
                    })
        
        comp_df = pd.DataFrame(comparison_data)
        
        st.write("**Average Fare by Weather:**")
        weather_avg = comp_df.groupby('Weather')['Fare'].mean().sort_values(ascending=False)
        st.bar_chart(weather_avg)
        
        st.write("**Average Fare by Traffic:**")
        traffic_avg = comp_df.groupby('Traffic')['Fare'].mean().sort_values(ascending=False)
        st.bar_chart(traffic_avg)
        
        st.write("**Average Fare by Hour:**")
        hour_avg = comp_df.groupby('Hour')['Fare'].mean().sort_values()
        st.line_chart(hour_avg)
        
        best_condition = comp_df.loc[comp_df['Fare'].idxmin()]
        worst_condition = comp_df.loc[comp_df['Fare'].idxmax()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("**Best Deal:**")
            st.write(f"Weather: {best_condition['Weather']}")
            st.write(f"Traffic: {best_condition['Traffic']}")
            st.write(f"Hour: {int(best_condition['Hour'])}:00")
            st.write(f"Fare: ${best_condition['Fare']:.2f}")
        
        with col2:
            st.error("**Most Expensive:**")
            st.write(f"Weather: {worst_condition['Weather']}")
            st.write(f"Traffic: {worst_condition['Traffic']}")
            st.write(f"Hour: {int(worst_condition['Hour'])}:00")
            st.write(f"Fare: ${worst_condition['Fare']:.2f}")