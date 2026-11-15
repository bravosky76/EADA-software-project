
import joblib
import pandas as pd
import pytest
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "model.pkl"


@pytest.fixture(scope="module")
def model():
    """Load the trained model once for all tests."""
    return joblib.load(MODEL_PATH)


def make_listing(guests=2, bedrooms=1):
    """Create a typical Barcelona Airbnb listing."""
    return pd.DataFrame([{
        "neighbourhood": "Eixample",
        "room_type": "Entire home/apt",
        "accommodates": guests,
        "bedrooms": bedrooms,
        "bathrooms": 1,
        "beds": 1 if guests == 2 else 3,
        "review_scores_rating": 4.8,
        "number_of_reviews": 50,
    }])


def test_model_loads(model):
    """Check the saved model loads successfully."""
    assert model is not None
    assert hasattr(model, "predict")


def test_prediction_is_positive(model):
    """A normal listing should have a positive price."""
    listing = make_listing()
    prediction = model.predict(listing)[0]

    assert prediction > 0


def test_larger_listing_costs_more(model):
    """A larger property should have a higher prediction."""
    small = make_listing(guests=2, bedrooms=1)
    large = make_listing(guests=6, bedrooms=3)

    small_price = model.predict(small)[0]
    large_price = model.predict(large)[0]

    assert large_price > small_price
