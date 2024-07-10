import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from geopy.geocoders import Nominatim

# Load the model
with open('model_rf.pkl', 'rb') as file:
    model_rf = pickle.load(file)

# Get feature names from the model
feature_names = model_rf.feature_names_in_  # Get the feature names used during training

# Function to extract features from the form data
def extract_features(data):
    features = {
        'Delivery_person_Age': data['Delivery_person_Age'],
        'Delivery_person_Ratings': data['Delivery_person_Ratings'],
        'Weather_conditions': data['Weather_conditions'],
        'Road_traffic_density': data['Road_traffic_density'],
        'Type_of_order': data['Type_of_order'],
        'Type_of_vehicle': data['Type_of_vehicle'],
        'multiple_deliveries': data['multiple_deliveries'],
        'Festival': data['Festival'],
        'City': data['City'],
        'Vehicle_condition': data['Vehicle_condition'],
        'day': data['day'],
        'month': data['month'],
        'quarter': data['quarter'],
        'year': data['year'],
        'day_of_week': data['day_of_week'],
        'is_weekend': data['is_weekend'],
        'order_prepare_time': data['order_prepare_time'],
        'distance': data['distance'],
        'prepare_time_per_km': data['prepare_time_per_km']
    }
    df_features = pd.DataFrame([features])
    df_features = df_features[feature_names]  # Reorder columns to match the model's expected order
    return df_features

# Function to format time as hours and minutes
def format_time(minutes):
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours} hours and {mins} minutes"

# Streamlit app
st.set_page_config(page_title="Food Delivery Time Predictor - From Order to Arrival", layout="wide", initial_sidebar_state="expanded")

# Define CSS for styling
st.markdown("""
    <style>
    .header {
        text-align: center;
        padding: 2rem;
    }
    .header h1 {
        font-size: 3rem;
        color: #2e86c1;
    }
    .header img {
        max-width: 100%;
        height: auto;
    }
    .form-container {
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
    }
    .form-container .col {
        width: 45%;
        padding: 1rem;
        box-sizing: border-box;
    }
    .gif-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    .gif-container img {
        max-width: 100%;
        height: auto;
    }
    .intro {
        font-size: 1.2rem;
        color: #555;
    }
    .custom-submit-button {
        background-color: #2e86c1;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        cursor: pointer;
        text-align: center;
    }
    .custom-submit-button:hover {
        background-color: #1d4f7a;
    }
    </style>
""", unsafe_allow_html=True)

# Home page with intro, GIF, and image
st.markdown('<div class="header"><h1>Food Delivery Time Predictor - From Order to Arrival!</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="gif-container"><img src="https://miro.medium.com/v2/resize:fit:600/1*fE3JkGyzhWXlXApVnShDtw.gif" alt="Food Delivery GIF"></div>', unsafe_allow_html=True)
st.markdown('<div class="intro" style="color:red;"><p>Welcome to the Food Delivery Time Prediction app! Here, you can predict the time it will take for a food delivery based on various factors.</p></div>', unsafe_allow_html=True)

# Option menu
st.sidebar.title(":rainbow[Menu]")
option = st.sidebar.selectbox("Select an Option", ["Get Prediction", "About"])

if option == "Get Prediction":
    # Form for user input
    with st.form(key='delivery_form'):
        st.subheader("Enter Delivery Details")
        col1, col2 = st.columns(2)

        with col1:
            Delivery_person_Age = st.selectbox("Delivery Person's Age", list(range(18, 51)))
            Delivery_person_Ratings = st.selectbox("Delivery Person's Ratings (1 to 5)", [1.0, 2.0, 3.0, 4.0, 5.0])
            Weather_conditions = st.selectbox("Weather Conditions", [1, 2, 3, 4, 5, 6, 7], format_func=lambda x: {1: 'Clear', 2: 'Windy', 3: 'Stormy', 4: 'Foggy', 5: 'Sunny', 6: 'Cloudy', 7: 'Sandstorms'}[x])
            Road_traffic_density = st.selectbox("Road Traffic Density", [1, 2, 3, 4], format_func=lambda x: {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}[x])
            Type_of_order = st.selectbox("Type of Order", [1, 2, 3, 4], format_func=lambda x: {1: 'Meal', 2: 'Drinks', 3: 'Snack', 4: 'Buffet'}[x])
            Type_of_vehicle = st.selectbox("Type of Vehicle", [1, 2, 3], format_func=lambda x: {1: 'Electric_scooter', 2: 'Motorcycle', 3: 'Scooter'}[x])
            Vehicle_condition = st.selectbox("Vehicle Condition", [1, 2, 3], format_func=lambda x: {1: 'New', 2: 'Old', 3: 'Very Old'}[x])

        with col2:
            # Single Date Picker
            selected_date = st.date_input("Select Date", value=pd.to_datetime('today'))
            day = selected_date.day
            month = selected_date.month
            year = selected_date.year
            quarter = (month - 1) // 3 + 1
            day_of_week = selected_date.weekday()
            is_weekend = 1 if selected_date.weekday() >= 5 else 0

            st.write(f"**Day:** {day}, **Month:** {month}, **Year:** {year}, /n **Quarter:** Q{quarter}, **Day of the Week:** {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week]}, **Is Weekend:** {'Yes' if is_weekend else 'No'}")
            
            multiple_deliveries = st.selectbox("Number of Deliveries", [0, 1, 2, 3])
            Festival = st.selectbox("Festival", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            City = st.selectbox("City", [1, 2, 3], format_func=lambda x: {1: 'Metropolitan', 2: 'Urban', 3: 'Semi-Urban'}[x])
            order_prepare_time = st.slider("Order Prepare Time (in minutes)", min_value=0, max_value=120, value=15, step=1)
            distance = st.slider("Distance (in km)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
            prepare_time_per_km = order_prepare_time / distance
            st.write(f"**Prepare Time Per Km (in minutes):** {prepare_time_per_km:.2f}")
            
        # Submit button with custom styling
        submit_button = st.form_submit_button(label='Get Prediction')  # Add the hidden submit button for functionality

        if submit_button:
            # Validate if all fields have been filled and have valid values
            if (Delivery_person_Age is not None and 18 <= Delivery_person_Age <= 50 and
                Delivery_person_Ratings is not None and 1.0 <= Delivery_person_Ratings <= 5.0 and
                Weather_conditions is not None and
                Road_traffic_density is not None and
                Type_of_order is not None and
                Type_of_vehicle is not None and
                Vehicle_condition is not None and
                multiple_deliveries is not None and
                Festival is not None and
                City is not None and
                day is not None and
                month is not None and
                quarter is not None and
                year is not None and
                day_of_week is not None and
                is_weekend is not None and
                order_prepare_time is not None and 0 <= order_prepare_time <= 120 and
                distance is not None and 0.1 <= distance <= 50.0 and
                prepare_time_per_km is not None and prepare_time_per_km > 0):

                # Extract features from the form data
                data = {
                    'Delivery_person_Age': Delivery_person_Age,
                    'Delivery_person_Ratings': Delivery_person_Ratings,
                    'Weather_conditions': Weather_conditions,
                    'Road_traffic_density': Road_traffic_density,
                    'Type_of_order': Type_of_order,
                    'Type_of_vehicle': Type_of_vehicle,
                    'Vehicle_condition': Vehicle_condition,
                    'multiple_deliveries': multiple_deliveries,
                    'Festival': Festival,
                    'City': City,
                    'day': day,
                    'month': month,
                    'quarter': quarter,
                    'year': year,
                    'day_of_week': day_of_week,
                    'is_weekend': is_weekend,
                    'order_prepare_time': order_prepare_time,
                    'distance': distance,
                    'prepare_time_per_km': prepare_time_per_km
                }

                features = extract_features(data)
                prediction = model_rf.predict(features)

                # Convert prediction to hours and minutes format
                predicted_time = format_time(prediction[0])
                
                st.subheader('Prediction Results')
                st.markdown(f"<h2 style='color: #E0F7FA;'>Predicted Delivery Time: {predicted_time}</h2>", unsafe_allow_html=True)

            else:
                # Display an error message if any field is invalid
                st.error(":red[Please ensure all fields are filled in correctly.]")

elif option == "About":
    st.title("About This App")
    st.markdown('<div class="gif-container"><img src="https://miro.medium.com/v2/resize:fit:600/1*SDQ9ly2pzx3UnGSS0L0WHQ.png" alt="Food Delivery Time Prediction Image"></div>', unsafe_allow_html=True)
    st.write("""
         This app predicts the delivery time for food orders based on various factors such as the age and ratings of the **delivery person, weather conditions, and traffic density**. The model used for predictions is a Random Forest Regressor trained on historical food delivery data.
    """)
    st.write("""
        ### Overview: 
          The food delivery industry has seen rapid growth in recent years,
          driven by advancements in technology and changing consumer habits. 
          However, one of the significant challenges within this domain is accurately predicting delivery times
          to ensure customer satisfaction and optimize delivery operations. """) 
    
    st.write("""
        ### How it Works:
        1. **Input Data**: You provide details about the delivery conditions.
        2. **Model Prediction**: The model processes this data and predicts the delivery time.
        3. **Results**: You receive the predicted delivery time based on the provided data.
    """)
