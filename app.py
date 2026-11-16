import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------- Visual style: animated Barcelona sunset background ----------
st.markdown("""
<style>
.stApp {background: linear-gradient(-45deg, #ff4d6d, #ff8fab, #ffd166, #e63946, #ff9e00); background-size: 400% 400%; animation: sunset 15s ease infinite;}
@keyframes sunset {0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;}}
.stApp::before, .stApp::after {content: ""; position: fixed; width: 420px; height: 420px; border-radius: 50%; filter: blur(90px); opacity: 0.55; z-index: 0; pointer-events: none;}
.stApp::before {background: #ffd166; top: -100px; left: -100px; animation: float1 18s ease-in-out infinite;}
.stApp::after {background: #ff006e; bottom: -120px; right: -120px; animation: float2 22s ease-in-out infinite;}
@keyframes float1 {0%, 100% {transform: translate(0, 0);} 50% {transform: translate(60vw, 40vh);}}
@keyframes float2 {0%, 100% {transform: translate(0, 0);} 50% {transform: translate(-55vw, -45vh);}}
[data-testid="stHeader"] {background: transparent;}
.block-container {position: relative; z-index: 1;}
h1 {color: #ffffff !important; font-weight: 800 !important; text-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);}
.block-container p {color: #ffffff;}
[data-testid="stForm"] {background: rgba(20, 10, 30, 0.45); backdrop-filter: blur(14px); border: 1px solid rgba(255, 255, 255, 0.35); border-radius: 22px; padding: 1.5rem; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25);}
[data-testid="stForm"] label p {color: #ffffff !important; font-weight: 600;}
[data-testid="stFormSubmitButton"] button {background: linear-gradient(90deg, #ff006e, #ff9e00); color: #ffffff; border: none; border-radius: 999px; padding: 0.6rem 2rem; font-weight: 700; transition: transform 0.15s;}
[data-testid="stFormSubmitButton"] button:hover {transform: scale(1.05);}
</style>
""", unsafe_allow_html=True)


st.title("Rip-off Radar BCN")
st.write("Compare an offered nightly price with the model's estimate.")

# Resolve files relative to this app, even when launched from another folder.
APP_DIR = Path(__file__).resolve().parent
try:
    model = joblib.load(APP_DIR / "model.pkl")
    listings = pd.read_csv(APP_DIR / "data" / "clean_listings.csv")
    neighbourhoods = sorted(listings["neighbourhood"].dropna().unique().tolist())
    if not neighbourhoods:
        raise ValueError("The listings file contains no neighbourhoods.")
except FileNotFoundError as error:
    st.error(f"Missing required file: {error.filename}. Please restore it and restart the app.")
    st.stop()
except Exception as error:
    st.error(f"Could not load the model or neighbourhood data: {error}")
    st.stop()

# A form waits until all details are entered before making a prediction.
with st.form("listing_details"):
    neighbourhood = st.selectbox("Neighbourhood", neighbourhoods)
    room_type = st.selectbox(
        "Room type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
    )
    accommodates = st.number_input("Accommodates", min_value=1, value=2, step=1)
    bedrooms = st.number_input("Bedrooms", min_value=0, value=1, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, value=1.0, step=0.5)
    beds = st.number_input("Beds", min_value=0, value=1, step=1)
    review_scores_rating = st.number_input(
        "Review scores rating (0–5)", min_value=0.0, max_value=5.0, value=4.5, step=0.1
    )
    number_of_reviews = st.number_input("Number of reviews", min_value=0, value=0, step=1)
    offered_price = st.number_input(
        "Offered nightly price (€)", min_value=0.0, value=100.0, step=1.0
    )
    submitted = st.form_submit_button("Check price")

if submitted:
    # These eight columns match train.py exactly. The offered price is not an input.
    features = pd.DataFrame([{
        "neighbourhood": neighbourhood,
        "room_type": room_type,
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "beds": beds,
        "review_scores_rating": review_scores_rating,
        "number_of_reviews": number_of_reviews,
    }])

    try:
        predictions = model.predict(features)
        if len(predictions) != 1:
            raise ValueError("Expected exactly one predicted price.")
        estimated_price = float(predictions[0])
        if not math.isfinite(estimated_price) or estimated_price <= 0:
            raise ValueError("The estimated price must be a finite, positive number.")
        if not math.isfinite(offered_price):
            raise ValueError("The offered price must be a finite number.")
    except Exception as error:
        st.error(f"Could not estimate a valid nightly price: {error}")
        st.stop()

    st.metric("Estimated nightly price", f"€{estimated_price:.2f}")
    # Strict comparisons keep exactly 85% and 115% in the fair range.
    if offered_price < 0.85 * estimated_price:
        st.success("Good deal")
    elif offered_price > 1.15 * estimated_price:
        st.error("Rip-off")
    else:
        st.warning("Fair price")
