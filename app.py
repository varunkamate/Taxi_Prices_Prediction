# Ultra-dynamic Streamlit app to load a pickled pipeline (best_model.pkl) and predict taxi prices.
# Save best_model.pkl in the same folder as this file and run:
# streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import sklearn

st.set_page_config(page_title="✨ Taxi Price Studio", layout="wide")

# ------------ Ultra-dynamic CSS (glass + animated gradient + neon) ------------
ultra_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;700&display=swap');
:root{
  --bg1:#0f172a; --bg2:#071038; --accent:#00f5ff; --glass: rgba(255,255,255,0.06);
}
body {
  background: linear-gradient(120deg, rgba(7,16,56,1) 0%, rgba(15,23,42,1) 50%, rgba(2,6,23,1) 100%);
  color: #e6eef8;
  font-family: Inter, sans-serif;
}
/* main container glass */
section.main > div.block-container{backdrop-filter: blur(8px); border-radius:18px; padding:26px; box-shadow: 0 8px 40px rgba(2,6,23,0.6);}
h1 { font-family: 'Orbitron', sans-serif; letter-spacing:1px; font-weight:700; color: white; }
.header-row{ display:flex; align-items:center; gap:18px; }
.logo-bubble{ width:86px; height:86px; border-radius:20px; background: linear-gradient(135deg,var(--accent),#7c3aed); display:flex; align-items:center; justify-content:center; box-shadow: 0 6px 30px rgba(124,58,237,0.25);}
.logo-bubble img{ width:60px; filter: drop-shadow(0 6px 20px rgba(0,0,0,0.6)); }
.card { background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:16px; padding:18px; border: 1px solid rgba(255,255,255,0.03);}
.btn-neon{ background: linear-gradient(90deg, #06b6d4, #7c3aed); color: white; border:none; padding:10px 16px; border-radius:12px; font-weight:700; }
.small-muted{ color: rgba(230,238,248,0.65); font-size:0.9rem }
.footer { text-align:center; font-size:0.85rem; color: rgba(230,238,248,0.6); margin-top:18px }
/* animated gradient underline */
.underline { height:6px; border-radius:999px; background: linear-gradient(90deg,#06b6d4,#7c3aed,#ff6ec7); background-size:200% 100%; animation: shimmer 4s linear infinite; }
@keyframes shimmer{ 0%{background-position:0% 50%} 100%{background-position:200% 50%} }
/* tweak Streamlit specific elements */
[data-testid='stToolbar']{display:none}
</style>
"""

st.markdown(ultra_css, unsafe_allow_html=True)

# ------------ Header ------------
col1, col2 = st.columns([1,4])
with col1:
    st.markdown("<div class='logo-bubble'><img src='https://img.icons8.com/ios-filled/100/ffffff/taxi.png' alt='taxi'/></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='header-row'><div><h1>Taxi Price Studio</h1><div class='small-muted'>Predict. Visualize. Iterate.</div></div></div>", unsafe_allow_html=True)

st.markdown("<div class='underline' style='margin-top:12px; margin-bottom:18px;'></div>", unsafe_allow_html=True)

# ------------ Load model ------------
@st.cache_resource
def load_model(path="taxi_pricing_best_model.pkl"):
    try:
        with open(path, "rb") as f:
            # The pickle file contains a tuple of (pipeline, sklearn_version)
            model, _ = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Failed to load model: The file '{path}' was not found.")
        return None
    except Exception as e:
        st.error(f"Failed to load model due to an unexpected error: {e}")
        return None

model = load_model("taxi_pricing_best_model.pkl")

# Define the expected columns from the training data
EXPECTED_COLS = ['Trip_Distance_km', 'Time_of_Day', 'Day_of_Week', 'Passenger_Count', 'Traffic_Conditions', 'Weather', 'Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes']

# Now, extract the categorical options from the loaded model's preprocessor
categorical_features = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather']
time_of_day_options = []
day_of_week_options = []
traffic_conditions_options = []
weather_options = []

if model is not None:
    try:
        # --- THIS IS THE FIX ---
        # Get the 'preprocessor' step from the pipeline
        preprocessor = model.named_steps['preprocessor']
        # Now access the 'named_transformers_' from the preprocessor
        categorical_transformer = preprocessor.named_transformers_['cat']
        categories = categorical_transformer.categories_

        time_of_day_options = list(categories[0])
        day_of_week_options = list(categories[1])
        traffic_conditions_options = list(categories[2])
        weather_options = list(categories[3])

    except KeyError as e:
        st.error(f"Error: Could not find the expected preprocessor step or transformer in the pipeline. Missing key: {e}")
        st.info("The pipeline steps might be named differently from 'preprocessor' and 'cat'.")
    except Exception as e:
        st.error(f"An unexpected error occurred while extracting categories from the model: {e}")

# If the model didn't load, show a warning
if model is None:
    st.warning("Please place 'taxi_pricing_best_model.pkl' in the same folder as this app and rerun.")
    st.stop() # Stop the app execution if the model is not available

# ------------ Sidebar: single input or batch ------------
st.sidebar.header("Input / Batch Options")
mode = st.sidebar.radio("Mode", ["Single prediction", "Batch upload (.csv)"])

# ------------ Main UI ------------
left_col, right_col = st.columns((2,1))
with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prepare input")

    if mode == "Single prediction":
        # show inputs form
        with st.form(key='single_pred'):
            inputs = {}
            inputs['Trip_Distance_km'] = st.number_input("Trip Distance (km)", min_value=0.0, value=5.0, step=0.1)
            inputs['Time_of_Day'] = st.selectbox("Time of Day", time_of_day_options)
            inputs['Day_of_Week'] = st.selectbox("Day of Week", day_of_week_options)
            inputs['Passenger_Count'] = st.number_input("Passenger Count", min_value=1, max_value=10, value=1, step=1)
            inputs['Traffic_Conditions'] = st.selectbox("Traffic Conditions", traffic_conditions_options)
            inputs['Weather'] = st.selectbox("Weather", weather_options)
            inputs['Base_Fare'] = st.number_input("Base Fare (currency)", min_value=0.0, value=3.0, step=0.5)
            inputs['Per_Km_Rate'] = st.number_input("Per Km Rate", min_value=0.0, value=0.8, step=0.01)
            inputs['Per_Minute_Rate'] = st.number_input("Per Minute Rate", min_value=0.0, value=0.2, step=0.01)
            inputs['Trip_Duration_Minutes'] = st.number_input("Trip Duration (minutes)", min_value=0.0, value=12.0, step=0.5)

            submit = st.form_submit_button("Run prediction", help="Predict trip price")

        if submit:
            X = pd.DataFrame([inputs], columns=EXPECTED_COLS)
            try:
                pred = model.predict(X)[0]
                st.markdown("<div class='card' style='margin-top:12px; padding:16px'>", unsafe_allow_html=True)
                st.markdown(f"## Predicted Trip Price: <span style='color:#00f5ff'>₹ {pred:,.2f}</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='small-muted'>Model: {model.__class__.__name__}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        st.markdown("Upload a CSV with columns exactly matching the training features.")
        uploaded = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded is not None:
            try:
                df_batch = pd.read_csv(uploaded)
                st.dataframe(df_batch.head())
                if st.button("Run batch predictions"):
                    try:
                        preds = model.predict(df_batch)
                        df_batch['Predicted_Trip_Price'] = preds
                        st.success("Batch prediction done")
                        st.dataframe(df_batch.head())

                        # allow download
                        csv = df_batch.to_csv(index=False).encode('utf-8')
                        b64 = base64.b64encode(csv).decode()
                        href = f"data:file/csv;base64,{b64}"
                        st.markdown(f"<a href='{href}' download='predicted_prices.csv' class='btn-neon' style='text-decoration:none; padding:10px 16px; display:inline-block;'>Download predictions</a>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Model & Diagnostics")
    st.write(f"**Pipeline:** {model.__class__.__name__}")
    # try to extract regressor
    try:
        # The last step of the pipeline is typically the model (regressor)
        regressor_model = model.named_steps[model.steps[-1][0]]

        if hasattr(regressor_model, 'feature_importances_'):
            # This applies to tree-based models like RandomForest
            st.markdown("**Feature importances**")
            # We can't easily get the feature names back, so we show the raw importances
            st.write(regressor_model.feature_importances_)
        elif hasattr(regressor_model, 'coef_'):
            # This applies to linear models like LinearRegression
            st.markdown("**Model coefficients**")
            st.write(regressor_model.coef_)
        else:
            st.info("Model does not expose feature importances or coefficients.")
    except Exception as e:
        st.info(f"Cannot extract diagnostics from pipeline: {e}")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>Built with ❤️ — Taxi Price Studio · Streamlit</div>", unsafe_allow_html=True)
