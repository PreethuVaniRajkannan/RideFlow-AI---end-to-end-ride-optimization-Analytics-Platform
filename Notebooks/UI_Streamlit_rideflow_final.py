import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import tensorflow as tf
from pytorch_tabnet.tab_model import TabNetRegressor
from transformers import pipeline
import keras
import torch
device = 0 if torch.cuda.is_available() else -1
# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="RideFlow AI", layout="wide")

st.title("🚗 RideFlow AI Platform")

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    keras.config.enable_unsafe_deserialization()
    models = {}

    # -------------------------------
    # Traditional ML Models
    # -------------------------------
    models['demand'] = pickle.load(open(
        r"C:\Users\chaka\Preethu\My_Git_Repo\Final Project_Rideflow\saved_models\Demand_pred_rfr_model.pkl", "rb"))

    models['supply'] = pickle.load(open(
        r"C:\Users\chaka\Preethu\My_Git_Repo\Final Project_Rideflow\saved_models\Supply_pred_rfr_model.pkl", "rb"))

    models['cancel'] = pickle.load(open(
        r"C:\Users\chaka\Preethu\My_Git_Repo\Final Project_Rideflow\saved_models\Cancellation_pred_rfc_model.pkl", "rb"))

    models['eta'] = pickle.load(open(
        r"C:\Users\chaka\Preethu\My_Git_Repo\Final Project_Rideflow\saved_models\XGB_ETA_prediction.pkl", "rb"))

    # -------------------------------
    # Deep Learning Model (Keras)
    # -------------------------------
    models['behavior'] = tf.keras.models.load_model(
        r"C:\Users\chaka\Preethu\My_Git_Repo\Final Project_Rideflow\saved_models\model_MobileNetV2.keras",
        compile=False
    )

    # -------------------------------
    # TabNet Model (FIXED)
    # -------------------------------
    tabnet = TabNetRegressor()

    tabnet.load_model(
        r"C:\Users\chaka\Preethu\My_Git_Repo\Final Project_Rideflow\saved_models\tabnet_hotspot.zip.zip"
    ) 

    models['tabnet'] = tabnet

    # -------------------------------
    # NLP Models
    # -------------------------------
    models['sentiment'] = pipeline(
        "sentiment-analysis",
        model=r"C:\Users\chaka\Preethu\My_Git_Repo\Final Project_Rideflow\saved_models\BERT_Feedback\models\sentiment_model",
        tokenizer=r"C:\Users\chaka\Preethu\My_Git_Repo\Final Project_Rideflow\saved_models\BERT_Feedback\models\sentiment_model",
        device=device
    )

    models['issue'] = pipeline(
    "zero-shot-classification",
    model=r"C:\Users\chaka\Preethu\My_Git_Repo\Final Project_Rideflow\saved_models\BERT_Feedback\models\issue_model",
    tokenizer=r"C:\Users\chaka\Preethu\My_Git_Repo\Final Project_Rideflow\saved_models\BERT_Feedback\models\issue_model",
    device=device,
    framework="pt" )

    return models

models = load_models()

# -------------------------------
# SURGE FUNCTION
# -------------------------------
def calculate_surge_advanced(demand, supply, base_fare,
                            traffic_level, weather, is_peak_hour):

    demand = max(0, demand)
    supply = max(1, supply)

    gap = demand - supply
    gap_ratio = gap / (supply + 1)

    surge = 1 + (0.5 * gap_ratio)

    if traffic_level >= 2:
        surge += 0.2

    if weather.lower() in ["rain", "storm"]:
        surge += 0.3

    if is_peak_hour == 1:
        surge += 0.2

    surge = max(0.8, min(surge, 3.0))
    final_price = base_fare * surge

    return round(surge, 2), round(final_price, 2)

#################################################################################################################
# =========================
# SIDEBAR
# =========================
page = st.sidebar.selectbox("Select Module", [
    "Dashboard",
    "Predictions",
    "Hotspot & Driver Behaviour",
    "AI Assistant"
])
#page1
# =========================
# DASHBOARD
# =========================
if page == "Dashboard":
    st.header("📊 Overview")
    st.write("Welcome to RideFlow AI system")
    st.caption("Dashboard provides real-time demand insights, peak hour detection, and trend visualization using ride-level data.")
    # -------------------------------
    # LOAD DATA
    # -------------------------------
    @st.cache_data
    def load_data():
        df = pd.read_csv(r"C:\Users\chaka\Preethu\My_Git_Repo\Final Project_Rideflow\Datasets\rideflow_datasets.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    df = load_data()
    
    # -------------------------------
    # FEATURE ENGINEERING
    # -------------------------------
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.dayofweek
    df['time_bin'] = df['timestamp'].dt.floor('30min')
    
    # Demand per time bin
    demand_df = df.groupby('time_bin').size().reset_index(name='demand')
    
    # Labeling
    low = demand_df['demand'].quantile(0.33)
    high = demand_df['demand'].quantile(0.66)
    
    def label(x):
        if x >= high:
            return "High"
        elif x >= low:
            return "Medium"
        else:
            return "Low"
    
    demand_df['level'] = demand_df['demand'].apply(label)
    # -------------------------------
    # METRICS (TOP CARDS)
    # -------------------------------
    total_rides = len(df)
    avg_demand = round(demand_df['demand'].mean(), 2)
    peak_hour = df['hour'].mode()[0]
    high_pct = round((demand_df['level'] == "High").mean() * 100, 2)
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Rides", total_rides)
    col2.metric("Avg Demand / Slot", avg_demand)
    col3.metric("Peak Hour", f"{peak_hour}:00")
    col4.metric("High Demand %", f"{high_pct}%")
    
    # -------------------------------
    # DEMAND TREND
    # -------------------------------
    st.subheader("📈 Demand Trend Over Time")
    
    fig_line = px.line(
        demand_df,
        x='time_bin',
        y='demand',
        title="Demand Trend"
    )
    
    st.plotly_chart(fig_line, use_container_width=True)
    
    # -------------------------------
    # DEMAND DISTRIBUTION
    # -------------------------------
    st.subheader("📊 Demand Distribution")
    
    dist = demand_df['level'].value_counts().reset_index()
    dist.columns = ['Demand Level', 'Count']
    
    fig_bar = px.bar(
        dist,
        x='Demand Level',
        y='Count',
        color='Demand Level',
        title="Demand Categories"
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # -------------------------------
    # PEAK HOURS ANALYSIS
    # -------------------------------
    st.subheader("⏱️ Demand by Hour")
    
    hourly = df.groupby('hour').size().reset_index(name='rides')
    
    fig_hour = px.bar(
        hourly,
        x='hour',
        y='rides',
        title="Rides by Hour"
    )
    
    st.plotly_chart(fig_hour, use_container_width=True)


    
########################################################################################################
#####################PAGE2##############################################################################
# =========================
#PREDICTIONS
# =========================
elif page == "Predictions":
    st.title("📊 Predicting Demand & Supply")
    # =========================
    #common inputs
    # =========================
    st.subheader("📥 Enter Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hour = st.slider("Hour", 0, 23, 18)
        day = st.selectbox("Day", list(range(7)),
                           format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])  
        pzone = st.selectbox("Pickup Zone", ["T Nagar", "Anna Nagar", "Tambaram", "Velachery", "OMR", "Adyar", "Porur"])
        
    with col2:
        surge = st.slider("Current Surge Multiplier", 1.0, 3.0, 1.2)
        traffic_str = st.selectbox("Traffic Level", ["Low", "Medium", "High"])   
        traffic = {
            "Low": 0,
            "Medium": 1,
            "High": 2
        }[traffic_str]
        weather = st.selectbox("Weather", ["clear", "cloudy", "rain"])
        base_fare = st.number_input("Base Fare (₹)", value=100.0)
        
    #=======================
    #Feature Engineering
    #=======================
    is_weekend = 1 if day in [5,6] else 0
    is_peak_hour = 1 if hour in [8,9,17,18,19] else 0
    
    
    # -------------------------------
    # ONE HOT ENCODING
    # -------------------------------
    data = {
        'hour': hour,
        'day_of_week': day,
        'is_weekend': is_weekend,
        'is_peak_hour': is_peak_hour,
        'surge_multiplier': surge,
        'traffic_level': traffic }
    #saving inputs
    st.session_state['user_inputs'] = data
    # Zones
    zones = ["T Nagar", "Anna Nagar", "Tambaram", "Velachery", "OMR", "Adyar", "Porur"]
    for z in zones:
        data[f'pickup_zone_{z}'] = 1 if z == pzone else 0
    
    # Weather
    weather_types = ["clear", "cloudy", "rain"]
    for w in weather_types:
        data[f'weather_{w}'] = 1 if w == weather else 0
   
    # -------------------------------
    # CREATE DF
    # -------------------------------
    input_df = pd.DataFrame([data])
    
    # Ensure correct column order (VERY IMPORTANT)
    feature_order = ['hour', 'day_of_week', 'is_weekend', 'is_peak_hour', 'surge_multiplier',
     'pickup_zone_T Nagar', 'pickup_zone_Anna Nagar', 'pickup_zone_Tambaram',
     'pickup_zone_Velachery', 'pickup_zone_OMR', 'pickup_zone_Adyar',
     'pickup_zone_Porur', 'traffic_level',
     'weather_clear', 'weather_cloudy', 'weather_rain']
    
    input_df = input_df[feature_order]
    # -------------------------------
    # PREDICT
    # -------------------------------
    if st.button("Predict"):
        demand = models['demand'].predict(input_df)[0] #demand
        # Add demand for supply model
        input_df['ride_count'] = demand
        # Step 3: Align features
        supply_features = models['supply'].feature_names_in_
        input_df = input_df.reindex(columns=supply_features, fill_value=0)
        supply = models['supply'].predict(input_df)[0] #supply
        # Surge
        surge, price = calculate_surge_advanced(
            demand,
            supply,
            base_fare,
            traffic,
            weather,
            is_peak_hour
        )
        # -------------------------------
        # OUTPUT
        # -------------------------------
        st.subheader("📊 Results")

        col1, col2, col3 = st.columns(3)
    
        col1.metric("📈 Demand", round(demand, 2))
        col2.metric("🚗 Supply", round(supply, 2))
        col3.metric("💰 Surge", f"{surge}x")
    
        st.metric("💵 Final Price", f"₹{price}")
    
        # Insight
        if demand > supply:
            st.error("🔥 High demand → Surge pricing applied")
        else:
            st.success("✅ Balanced market")
        st.session_state['predictions'] = {
                            "demand": demand,
                            "supply": supply,
                            "surge": surge,
                            "price": price}
    # =========================
    # ETA
    # =========================
    st.subheader("⏱ ETA Prediction")
    
    # =========================
    # ETA INPUTS
    # =========================
    col3, col4 = st.columns(2)
    
    with col3:
        distance = st.number_input("Distance (km)", min_value=0.0, value=5.0)
    
    with col4:
        dzone = st.selectbox(
            "Drop Zone",
            ["T Nagar", "Anna Nagar", "Tambaram", "Velachery", "OMR", "Adyar", "Porur"]
        )

    # =========================
    # PREDICT ETA
    # =========================
    if st.button("Predict ETA"):
        # 🚨 Check if predictions exist
        if 'predictions' not in st.session_state:
            st.warning("⚠️ Please run Demand & Supply prediction first")
            st.stop()
            # Load previous results
        demand = st.session_state['predictions']['demand']
        supply = st.session_state['predictions']['supply']
        surge = st.session_state['predictions']['surge']
        # Start from base input
        eta_data = input_df.copy()
    
        # Add model outputs
        eta_data['demand'] = demand
        eta_data['supply'] = supply
        eta_data['surge_multiplier'] = surge
    
        # Add new feature
        eta_data['distance'] = distance
    
        # -------------------------------
        # DROP ZONE ENCODING (ONE HOT)
        # -------------------------------
        zones = ["T Nagar", "Anna Nagar", "Tambaram", "Velachery", "OMR", "Adyar", "Porur"]
    
        for z in zones:
            eta_data[f'drop_zone_{z}'] = 1 if z == dzone else 0

        # -------------------------------
        # ALIGN FEATURES
        # -------------------------------
        eta_features = models['eta'].feature_names_in_
        eta_data = eta_data.reindex(columns=eta_features, fill_value=0)
    
        # -------------------------------
        # PREDICT
        # -------------------------------
        eta = models['eta'].predict(eta_data)[0]
    
        # =========================
        # OUTPUT
        # =========================
        st.success(f"⏱ Estimated ETA: {round(eta, 2)} mins")

        # SAVE FOR NEXT BLOCKS
        st.session_state['eta'] = eta
        st.session_state['distance'] = distance
            
    #===========================
    #Cancellation risk
    #===========================
    
    st.subheader("🚫 Cancellation Risk - Prediction")
            
    col5, col6 = st.columns(2)
            
    with col5:
        customer_rating = st.slider("Customer Rating", 1.0, 5.0, 4.2)
            
    with col6:
        driver_rating = st.slider("Driver Rating", 1.0, 5.0, 4.5)
            
        # =========================
        # PREDICT CANCELLATION
        # =========================
    if st.button("Predict Cancellation"):
    
        # 🚨 Ensure required data exists
        if 'predictions' not in st.session_state:
            st.warning("⚠️ Run Demand/Supply Prediction first")
            st.stop()
    
        if 'eta' not in st.session_state:
            st.warning("⚠️ Please predict ETA first")
            st.stop()
    
        # ✅ LOAD FROM SESSION
        price = st.session_state['predictions']['price']
        eta = st.session_state['eta']
        distance = st.session_state['distance']
    
        # -------------------------------
        # AUTO CALCULATED FEATURES
        # -------------------------------
        expected_eta = distance * 2
        eta_diff = eta - expected_eta
    
        cancel_data = pd.DataFrame([{
            'customer_rating': customer_rating,
            'fare_price': price,
            'eta_diff': eta_diff,
            'driver_rating': driver_rating,
            'distance_km': distance,
            'estimated_eta_min': eta,
            'hour': hour,
            'day_of_week': day,
            'traffic_level': traffic,
            'is_peak_hour': is_peak_hour
        }])
    
        # Weather encoding
        cancel_data['weather_clear'] = 1 if weather == "clear" else 0
        cancel_data['weather_cloudy'] = 1 if weather == "cloudy" else 0
        cancel_data['weather_rain'] = 1 if weather == "rain" else 0
    
        # Align features
        cancel_features = models['cancel'].feature_names_in_
        cancel_data = cancel_data.reindex(columns=cancel_features, fill_value=0)
    
        # Predict
        cancel_pred = models['cancel'].predict(cancel_data)[0]
    
        try:
            cancel_prob = models['cancel'].predict_proba(cancel_data)[0][1]
        except:
            cancel_prob = None
    
        # Output
        if cancel_pred == 1:
            st.error("🚫 High chance of Ride Cancellation")
        else:
            st.success("✅ Ride likely to be completed")
    
        if cancel_prob:
            st.metric("📊 Cancellation Probability", f"{round(cancel_prob*100,2)}%")
         # -------------------------------
        # SMART INSIGHTS 🔥
        # -------------------------------
        if cancel_pred == 1:
            if eta > 15:
                st.warning("⏱ Long ETA is increasing cancellation risk")
    
            if price > 300:
                st.warning("💸 High fare contributing to cancellation")
    
            if eta_diff > 5:
                st.warning("⚠️ ETA delay is a major factor")

########################################################################################################
#####################PAGE3##############################################################################
elif page == "Hotspot & Driver Behaviour":
    st.title("📍 Hotspot & Driver Behaviour Intelligence")
    # =========================
    # HOTSPOT PREDICTION
    # =========================
    st.subheader("🔥 Hotspot Prediction")

    col1, col2 = st.columns(2) 

    with col1:
        hour = st.slider("Hour", 0, 23, 18)
        day = st.selectbox("Day", list(range(7)),
                           format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

    with col2:
        traffic_str = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
        traffic = {"Low": 0, "Medium": 1, "High": 2}[traffic_str]

        weather = st.selectbox("Weather", ["clear", "cloudy", "rain"])

    # Feature Engineering
    is_weekend = 1 if day in [5,6] else 0
    is_peak_hour = 1 if hour in [8,9,17,18,19] else 0

    # Build input for TabNet
    hotspot_data = pd.DataFrame([{
        'hour': hour,
        'day_of_week': day,
        'traffic_level': traffic,
        'is_weekend': is_weekend,
        'is_peak_hour': is_peak_hour,
        'weather_clear': 1 if weather == "clear" else 0,
        'weather_cloudy': 1 if weather == "cloudy" else 0,
        'weather_rain': 1 if weather == "rain" else 0
    }])

    # Align features
    hotspot_features = models['tabnet'].feature_names
    hotspot_data = hotspot_data.reindex(columns=hotspot_features, fill_value=0)

    if st.button("Predict Hotspot"):

        hotspot_pred = models['tabnet'].predict(hotspot_data)[0]

        st.success(f"📍 Predicted Ride Demand Zone Score: {round(float(hotspot_pred),2)}")

        # Insight
        if hotspot_pred > 0.7:
            st.error("🔥 High Demand Hotspot → Deploy more drivers")
        elif hotspot_pred > 0.4:
            st.warning("⚠️ Moderate demand area")
        else:
            st.info("✅ Low demand zone")

    # =========================
    # DRIVER BEHAVIOUR
    # =========================
    st.subheader("🧠 Driver Behaviour Detection")

    uploaded_file = st.file_uploader("Upload Driver Image", type=["jpg","png","jpeg"])

    if uploaded_file is not None:

        # Display image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Preprocess Image
        from PIL import Image
        img = Image.open(uploaded_file).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("Analyze Behaviour"):

            prediction = models['behavior'].predict(img_array)[0][0]

            # Output
            if prediction > 0.5:
                st.error("🚫 Unsafe Driving Detected")
            else:
                st.success("✅ Safe Driving Behaviour")

            # Confidence
            st.metric("Confidence Score", f"{round(float(prediction*100),2)}%")

            # Insight
            if prediction > 0.7:
                st.warning("⚠️ High risk driver → consider blocking rides")

