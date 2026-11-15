import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


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
