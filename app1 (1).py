import streamlit as st
import pickle
import numpy as np
import base64 # to set the backgroud of web page 

# Load the model
with open("dtr.pkl", "rb") as file:
    model = pickle.load(file) 

# Set page configuration
st.set_page_config(
    page_title="Solar Power Prediction",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Set background image
def set_background(image_path):
    # Read the image and convert to Base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    
    page_bg = f"""
    <style>
    .stApp {{
        background: url(data:image/jpeg;base64,{encoded_image});
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}SdS
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# Set the background with your image
set_background("C:/Users/HP/Pictures/Wallpaper/solar-panel-belt-conveyor.jpeg")

# Add custom CSS for title, button, input field animations, and success message
st.markdown("""
    <style>
    .title {
        font-size: 3em;
        color: #FFD700; 
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
        padding: 10px;
        border: 3px solid #000000;
        border-radius: 10px; 
    }

    /* RGB Animation for Predict Button */
    .stButton > button {S
        background: linear-gradient(90deg, red, orange, yellow, green, cyan, blue, violet);
        background-size: 400% 400%;
        animation: gradientBG 3s infinite;
        color: white;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: transform 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.1); /* Button grows slightly */
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Custom styling for labels and input fields */
    .stNumberInput label {
        color: #F8F8FF; /* Light whitening color for labels */
        font-size: 1.2em; /* Larger font size for labels */
    }

    input[type=number] {
        border: 2px solid #FFD700;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
        transition: box-shadow 0.3s, border-color 0.3s;
        color: #F8F8FF; /* Light whitening color for text inside the input fields */
        background-color: #1E1E1E; /* Dark background for contrast */
    }

    input[type=number]:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 10px #4CAF50;
        outline: none;
    }

    /* Lighting yellow effect for success message */
    .success-message {
        color: #FFD700; /* Yellow text color */
        text-shadow: 0 0 2px #FFD700, 0 0 5px #FFD700, 0 0 15px #FFD700, 0 0 25px #FFD700;
        -webkit-text-stroke: 1px black; /* Adds a black border */
        text-stroke: 1px black; 
        font-size: 1.8em;
        font-weight: bold;
        margin-top: 20px;
    }

    </style>
""", unsafe_allow_html=True)

# Display title
st.markdown('<div class="title">Solar Power Generation</div>', unsafe_allow_html=True)

# User inputs
distance_to_solar_noon = st.number_input("Distance to Solar Noon", value=0.0)
temperature = st.number_input("Temperature (°C)", value=0.0)
wind_direction = st.number_input("Wind Direction (°)", value=0.0)
wind_speed = st.number_input("Wind Speed (m/s)", value=0.0)
sky_cover = st.number_input("Sky Cover (%)", value=0.0)
humidity = st.number_input("Humidity (%)", value=0.0)
average_wind_speed = st.number_input("Average Wind Speed (m/s)", value=0.0)
average_pressure = st.number_input("Average Pressure (hPa)", value=0.0)

# Combine all inputs
all_inputs = np.array([[ 
    distance_to_solar_noon, 
    temperature, 
    wind_direction, 
    wind_speed, 
    sky_cover, 
    humidity, 
    average_wind_speed, 
    average_pressure 
]])

# Predict button
if st.button("Predict"):
    # Prediction
    prediction = model.predict(all_inputs)[0]
    st.markdown(f'<div class="success-message">The predicted power generation is: {round(prediction, 2)} Watts</div>', unsafe_allow_html=True)
    