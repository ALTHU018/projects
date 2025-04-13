import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the trained Random Forest model
model = pickle.load(open("C:\\Users\\althu shaik\\Downloads\\final_model (1).pkl", "rb"))

# Streamlit UI
st.title("📈 Apple Stock Price Predictor")

# Sidebar Inputs
nasdaq_index = st.sidebar.number_input("NASDAQ Index", min_value=0, max_value=610000, value=15000)
sp500_index = st.sidebar.number_input("S&P 500 Index", min_value=0, max_value=250000, value=4000)
inflation_rate = st.sidebar.slider("Inflation Rate (%)", 0.0, 5.0, 3.0)
unemployment_rate = st.sidebar.slider("Unemployment Rate (%)", 0.0, 7.0, 5.0)
interest_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 7.0, 4.0)
market_sentiment = st.sidebar.slider("Market Sentiment (-1 to 1)", -1.0, 1.0, 0.5)

# Prepare input (Without Scaling)
input_data = np.array([[nasdaq_index, sp500_index, inflation_rate, unemployment_rate, interest_rate, market_sentiment]])

# Predict Next 30 Days
if st.sidebar.button("🔮 Predict Next 30 Days"):
    predictions = []
    current_input = input_data.copy()  # No scaling applied

    for _ in range(30):
        next_price = model.predict(current_input)[0]
        predictions.append(next_price)

        # Simulating small daily market variations
        current_input[0, 0] *= np.random.uniform(0.99, 1.01)  # NASDAQ variation
        current_input[0, 1] *= np.random.uniform(0.99, 1.01)  # S&P 500 variation
        current_input[0, 2] *= np.random.uniform(0.99, 1.01)  # Inflation change
        current_input[0, 3] *= np.random.uniform(0.99, 1.01)  # Unemployment change
        current_input[0, 4] *= np.random.uniform(0.99, 1.01)  # Interest rate fluctuation
        current_input[0, 5] *= np.random.uniform(0.99, 1.01)  # Market sentiment change

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, 31), predictions, marker='o', linestyle='-', color='blue', label="Predicted Prices")
    ax.set_xlabel("Days")
    ax.set_ylabel("Stock Price ($)")
    ax.set_title("Apple Stock Price Prediction for Next 30 Days")
    ax.legend()
    st.pyplot(fig)

    st.success(f"📌 **Predicted Stock Price after 30 days: ${predictions[-1]:.2f}**")
