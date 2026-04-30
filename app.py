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
import plotly.express as px
device = 0 if torch.cuda.is_available() else -1
from PIL import Image
from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="RideFlow AI", layout="wide")

st.title("🚗 RideFlow AI Platform")

# =========================
# SAFE SESSION INIT
# =========================
def init_session():
    defaults = {
        "inputs": {},
        "predictions": {},
        "eta": None,
        "distance": None
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

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
    "Hotspot Detection",
    "Driver Behaviour",
    "Feedback Analysis",
    "AI Assistant"
])
#######page1#############################################################################################################
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
        demand = st.session_state.get('predictions')['demand']
        supply = st.session_state.get('predictions')['supply']
        surge = st.session_state.get('predictions')['surge']
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
    
        # LOAD FROM SESSION
        price = st.session_state.get('predictions')['price']
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
elif page == "Hotspot Detection":
    st.title("📍 Hotspot Detection")
    # =========================
    # HOTSPOT PREDICTION
    # =========================
    if not st.session_state.get('predictions'):
        st.warning("⚠️ Please run Predictions first")
        st.stop()
    # =========================
    # NEW USER INPUTS
    # =========================
    col1, col2 = st.columns(2) 
    with col1:
        hour = st.slider("Hour", 0, 23, 18)
        day = st.selectbox("Day", list(range(7)),
                           format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        month = st.number_input("Month", 1, 12, 6)

    with col2:
        selected_zone = st.selectbox("Focus Zone (optional)", 
        ["All", "T Nagar", "Anna Nagar", "Tambaram", "Velachery", "OMR", "Adyar", "Porur"]
    )

    #inputs
    inputs = st.session_state['inputs'].copy()
    preds = st.session_state.get("predictions")
    demand = preds['demand']

    #hotspot function
    def get_hotspot_map_data(models, hour, day, month, demand):

        zone_coords = {
            "T Nagar": (13.0418, 80.2337),
            "Anna Nagar": (13.0850, 80.2101),
            "Tambaram": (12.9249, 80.1000),
            "Velachery": (12.9815, 80.2180),
            "OMR": (12.9170, 80.2300),
            "Adyar": (13.0012, 80.2565),
            "Porur": (13.0356, 80.1588)
        }
    
        zone_map = {
            "T Nagar": 0, "Anna Nagar": 1, "Tambaram": 2,
            "Velachery": 3, "OMR": 4, "Adyar": 5, "Porur": 6
        }
    
        rows = []
    
        for zone, (lat, lon) in zone_coords.items():
            row = {
                'pickup_lat': lat,
                'pickup_long': lon,
                'hour': hour,
                'day': day,
                'month': month,
                'demand': demand,
                'is_weekend': 1 if day in [5,6] else 0,
                'peak_hour': 1 if hour in [8,9,17,18,19] else 0,
                'zone': zone_map[zone]
            }
    
            rows.append(row)
    
        df = pd.DataFrame(rows)
    
        arr = df.astype(np.float32).values

        if len(arr.shape) == 1:
            arr = arr.reshape(1, -1)
        
        preds = models['tabnet'].predict(arr)
        
        preds = preds.flatten()
        
        # Fix mismatch
        if len(preds) != len(df):
            preds = preds[:len(df)]
        
        df['score'] = preds
        return df
     # =========================
    # CLASSIFICATION FUNCTION
    # =========================
    def classify(score):
        if score >= 0.75:
            return "High"
        elif score >= 0.5:
            return "Medium"
        else:
            return "Low"
   
    # =========================
    # PREDICT
    # =========================

    if st.button("Predict Hotspot"):
        hotspot_df = get_hotspot_map_data(models, hour, day, month, demand)

        # Optional filter for selected zone
        zone_map = {
            "T Nagar": 0, "Anna Nagar": 1, "Tambaram": 2,
            "Velachery": 3, "OMR": 4, "Adyar": 5, "Porur": 6
        }

        if selected_zone != "All":
            hotspot_df = hotspot_df[
                hotspot_df['zone'] == zone_map[selected_zone]
            ]

        if hotspot_df.empty:
            st.error("❌ No data to display")
            st.stop()
        # =========================
        # ADD LABELS
        # =========================
        zone_names = {
            0:"T Nagar", 1:"Anna Nagar", 2:"Tambaram",
            3:"Velachery", 4:"OMR", 5:"Adyar", 6:"Porur"
        }

        hotspot_df['zone_name'] = hotspot_df['zone'].map(zone_names)
        hotspot_df['level'] = hotspot_df['score'].apply(classify)
        # =========================
        # SCORE
        # =========================
        overall_score = float(hotspot_df['score'].mean())
        overall_level = classify(overall_score)
        
        # Insight
        if overall_level == "High":
            st.error("🔥 High Demand Hotspot → Deploy more drivers")
        elif overall_level == "Medium":
            st.warning("⚠️ Moderate demand area")
        else:
            st.info("✅ Low demand zone")
        
    
       # =========================
        # MAP
        # =========================
        st.subheader("🗺️ Hotspot Map")
        color_map = {
            "Low": "green",
            "Medium": "yellow",
            "High": "red"
        }
         # Bubble size (always visible)
        hotspot_df['bubble_size'] = np.abs(hotspot_df['score']) * 5 + 10
        
        fig = px.scatter_mapbox(
            hotspot_df,
            lat="pickup_lat",
            lon="pickup_long",
            color="level",
            size="bubble_size",
            hover_name="zone_name",
            color_discrete_map=color_map,
            size_max=35,
            zoom=10,
            height=500
        )
        fig.update_layout(mapbox_style="open-street-map")
    
        st.plotly_chart(fig, use_container_width=True)
        # =========================
        # TOP HOTSPOTS
        # =========================
        st.subheader("🏆 Top Hotspots")

        top_df = hotspot_df.sort_values("score", ascending=False)[
            ['zone_name', 'score', 'level']
        ]

        st.dataframe(top_df)
######################################################################################################
########page4#########################################################################################

elif page == "Driver Behaviour":
    st.subheader("🧠 Driver Behaviour Detection")

    uploaded_file = st.file_uploader("Upload Driver Image", type=["jpg", "png", "jpeg"], accept_multiple_files=False)
    st.write("File:", uploaded_file)
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB") 
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # =========================
        # PREPROCESS IMAGE
        # ========================= 
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # =========================
        # PREDICTION
        # =========================
    if st.button("Analyze Behaviour"):

        pred = models['behavior'].predict(img_array)
        preds = pred[0]
    
        # =========================
        # ORIGINAL MODEL CLASSES
        # =========================
        class_labels = [
            "distractions",
            "safe_driving",
            "talking_phone",
            "texting_phone",
            "turning"
        ]
    
        class_index = np.argmax(preds)
        confidence = float(preds[class_index])
        predicted_class = class_labels[class_index]
    
        # =========================
        # MAP TO FINAL CLASSES
        # =========================
        if predicted_class in ["distractions", "turning"]:
            final_class = "Distractions"
            st.error(f"🚫 {final_class}")
    
        elif predicted_class in ["talking_phone", "texting_phone"]:
            final_class = "Phone Usage"
            st.warning(f"📱 {final_class}")
    
        else:
            final_class = "Safe Driving"
            st.success(f"✅ {final_class}")
    
        # =========================
        # CONFIDENCE
        # =========================
        st.metric("Confidence Score", f"{round(confidence * 100, 2)}%")
    
        # =========================
        # INSIGHTS
        # =========================
        if final_class == "Distractions":
            st.error("🚨 High risk behaviour → restrict driver or alert system")
    
        elif final_class == "Phone Usage":
            st.warning("⚠️ Driver using phone → unsafe, monitor closely")
    
        else:
            st.info("✅ Driver behaviour is safe")
#==========================================Page5=================================================
#AI Assistant 
#================================================================================================
elif page == "AI Assistant":
    st.title("🚖 Ride Matching Assistant")
    # =========================
    # LOAD DATA
    # =========================
    data = pd.read_csv(r"C:\Users\chaka\Preethu\My_Git_Repo\Final Project_Rideflow\Datasets\preprocessed_rideflow_datasets.csv")

    driver_data = data[['driver_id', 'driver_rating', 'cancellation_risk', 'estimated_eta_min']].dropna()

    # =========================
    # GET DEMAND FROM PREVIOUS MODEL
    # =========================
    if not st.session_state.get('predictions'):
        st.warning("⚠️ Please run Predictions first")
        st.stop()

    demand_value = st.session_state['predictions']['demand']

    # Convert numeric demand → category
    if demand_value > 0.7:
        demand = "High"
    elif demand_value > 0.4:
        demand = "Medium"
    else:
        demand = "Low"

    st.info(f"📊 Current Demand Level: {demand}")

    # Convert to dict
    drivers = driver_data.to_dict(orient='records')

    # =========================
    # SCORING FUNCTION
    # =========================
    def compute_score(driver, demand):
        eta = float(driver["estimated_eta_min"])
        rating = float(driver["driver_rating"])
        cancel = float(driver["cancellation_risk"])

        eta_score = 1 / (eta + 1)
        rating_score = rating / 5
        cancel_score = 1 - cancel

        if demand == "High":
            score = (0.5 * eta_score) + (0.3 * rating_score) + (0.2 * cancel_score)
        elif demand == "Medium":
            score = (0.3 * eta_score) + (0.4 * rating_score) + (0.3 * cancel_score)
        else:
            score = (0.2 * eta_score) + (0.5 * rating_score) + (0.3 * cancel_score)

        return float(score)

    # =========================
    # RECOMMEND DRIVER
    # =========================
    def recommend_driver(drivers, demand):
        for d in drivers:
            d["score"] = compute_score(d, demand)

        sorted_drivers = sorted(drivers, key=lambda x: x["score"], reverse=True)
        return sorted_drivers[0], sorted_drivers

    # =========================
    # EXPLANATION
    # =========================
    def explain_recommendation(driver, demand):
        return f"""
🚗 Recommended Driver: {driver['driver_id']}

Reason:
- ETA: {round(driver['estimated_eta_min'],2)} mins (optimized for {demand} demand)
- Rating: {round(driver['driver_rating'],2)}
- Cancellation Risk: {round(driver['cancellation_risk'],2)}

✅ Best balance of speed, reliability, and service quality.
"""

    # =========================
    # BUTTON ACTION
    # =========================
    if st.button("Find Best Driver"):

        best_driver, ranked_drivers = recommend_driver(drivers, demand)

        # =========================
        # BEST DRIVER OUTPUT
        # =========================
        st.success(f"🚖 Best Driver: {best_driver['driver_id']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("⭐ Rating", round(best_driver['driver_rating'], 2))
        col2.metric("⏱ ETA", round(best_driver['estimated_eta_min'], 2))
        col3.metric("❌ Cancel Risk", round(best_driver['cancellation_risk'], 2))

        # Explanation
        st.subheader("🧠 AI Explanation")
        st.write(explain_recommendation(best_driver, demand))

        # =========================
        # TOP DRIVERS TABLE
        # =========================
        ranked_df = pd.DataFrame(ranked_drivers)

        st.subheader("🏆 Top Driver Rankings")
        st.dataframe(
            ranked_df[['driver_id', 'score', 'driver_rating', 'estimated_eta_min']]
            .sort_values(by="score", ascending=False)
            .head(10))
    #=======================
    #Chatbot
    #=======================
    st.subheader("💬 Chatbot")
     # Take one ride sample (or connect with session predictions)
    ride = data.iloc[0]

    translator = Translator()

    # =========================
    # INIT CHAT HISTORY
    # =========================
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    preds = st.session_state['predictions']
    eta = st.session_state.get('eta')
    distance = st.session_state.get('distance')

    demand = preds.get('demand')
    supply = preds.get('supply')
    surge = preds.get('surge')
    price = preds.get('price')

    # =========================
    # INTENT DETECTION
    # =========================
    def detect_intent(text):
        text = text.lower()

        if "where" in text or "driver" in text:
            return "track_driver"
        elif "eta" in text or "arrive" in text:
            return "eta"
        elif "cancel" in text:
            return "cancel"
        elif "price" in text or "fare" in text:
            return "price"
        elif "demand" in text or "availability" in text:
            return "demand"
        elif "hotspot" in text or "zone" in text:
            return "best zone"
        else:
            return "general"

    # =========================
    # RESPONSE GENERATOR
    # =========================
    def generate_response(intent, ride):
        if intent == "track_driver":
            return f"Your driver is on the way and will arrive in {round(ride['estimated_eta_min'],2)} minutes."
        
        elif intent == "eta":
            return f"Estimated arrival time is {round(ride['estimated_eta_min'],2)} minutes."
        
        elif intent == "cancel":
            if eta and eta > 15:
                return "High chance of cancellation due to long ETA."
            elif surge > 1.5:
                return "High fare may lead to cancellations."
            else:
                return "Cancellation risk is low."
        
        elif intent == "price":
            return f"Estimated fare is ₹{round(price,2)}"
        
        elif intent == "demand":
            if demand > supply:
                return "Demand is high currently. Expect surge pricing."
            else:
                return "Demand is stable."
        elif intent == "best zone":
            return"Check Hotspot module for best zones."
        else:
            return "How can I assist you with your ride?"

    # =========================
    # TRANSLATION FUNCTION
    # =========================
    def translate_text(text, lang):
        try:
            return translator.translate(text, dest=lang).text
        except:
            return text  # fallback

    # =========================
    # LANGUAGE SELECTOR
    # =========================
    lang = st.selectbox("🌐 Select Language", ["en", "hi", "ta"])

    # =========================
    # CHAT INPUT
    # =========================
    user_input = st.chat_input("Type your message...")

    if user_input:

        # Save user message
        st.session_state.chat_history.append(("You", user_input))

        # Process
        intent = detect_intent(user_input)
        response = generate_response(intent, ride)
        translated_response = translate_text(response, lang)

        # Save bot response
        st.session_state.chat_history.append(("Bot", translated_response))

    # =========================
    # DISPLAY CHAT
    # =========================
    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            with st.chat_message("user"):
                st.write(msg)
        else:
            with st.chat_message("assistant"):
                st.write(msg)
##################################################################################################################
elif page == "Feedback Analysis":
    st.title("💬 Feedback Sentiment Analysis (BERT)")
    # =========================
    # INPUT
    # =========================
    feedback = st.text_area("✍️ Enter customer feedback:")

    if st.button("Analyze"):
    
        if feedback.strip() == "":
            st.warning("Enter feedback")
            st.stop()
    
        # =========================
        # SENTIMENT
        # =========================
        sent_result = models['sentiment'](feedback)[0]
    
        sentiment = sent_result['label']
        sent_score = sent_result['score']
    
        # =========================
        # ISSUE (ZERO-SHOT)
        # =========================
        candidate_labels = [
            "Driver Behavior",
            "Delay / ETA Issue",
            "Pricing Issue",
            "App Bug",
            "Payment Issue"
        ]
    
        issue_result = models['issue'](
            feedback,
            candidate_labels=candidate_labels
        )
    
        issue = issue_result['labels'][0]
        issue_score = issue_result['scores'][0]
    
        # =========================
        # OUTPUT
        # =========================
        col1, col2 = st.columns(2)
    
        with col1:
            st.subheader("Sentiment")
            st.write(sentiment)
            st.metric("Confidence", f"{round(sent_score*100,2)}%")
    
        with col2:
            st.subheader("Issue Type")
            st.write(issue)
            st.metric("Confidence", f"{round(issue_score*100,2)}%")
    
       
