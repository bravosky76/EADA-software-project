# Rip-off Radar BCN

**Find out whether a Barcelona Airbnb is a good deal, fairly priced, or a rip-off.**

Rip-off Radar BCN is a Streamlit web application that uses machine learning to estimate the fair nightly price of an Airbnb listing in Barcelona.

## The Problem

Airbnb prices can vary significantly depending on location, property size, guest capacity, and other features. It can be difficult for travellers to know whether a listing is reasonably priced.

Rip-off Radar BCN helps users make more informed booking decisions by comparing a listing's advertised nightly price against a machine-learning prediction.

## How It Works

1. The app uses Barcelona Airbnb listing data from [Inside Airbnb](https://insideairbnb.com/).
2. The data is cleaned and prepared for machine learning.
3. A linear regression model is trained to predict nightly prices based on listing characteristics.
4. Users enter the details of an Airbnb listing into the Streamlit interface.
5. The app estimates the fair nightly price and compares it with the advertised price.
6. Based on the comparison, the listing is classified as a **Good Deal**, **Fair**, or **Rip-off**.

## Getting Started

### Requirements

- Python 3
- pip
- Project dependencies listed in `requirements.txt`

### Installation

Clone the repository and open the project folder.

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will display a local URL where you can access the application in your browser.

## Retraining the Model

To retrain the model using Inside Airbnb data:

1. Visit [Inside Airbnb](https://insideairbnb.com/) and download Barcelona's `listings.csv.gz` dataset.
2. Place `listings.csv.gz` inside the project's `data/` folder.
3. Run the data-cleaning script:

```bash
python clean_data.py
```

4. Train the linear regression model:

```bash
python train.py
```

The training script prepares the model for subsequent price predictions.

## Model Results

The linear regression model achieved the following reported performance:

| Metric | Result |
|---|---|
| R² Score | 0.371 |
| Average prediction error | Approximately €81 per night |

An R² score of 0.371 means the model explains approximately 37.1% of the variation in nightly prices in the evaluation data.

An average prediction error of approximately €81 per night indicates that predictions can differ substantially from actual listing prices.

These results make the app more suitable as a general price-comparison guide than an exact property valuation tool.

## Running Tests

The project uses pytest for automated testing.

Run:

```bash
pytest
```

The tests check whether the saved model loads, returns positive predictions for normal listings, and predicts a higher price for a larger listing in a representative comparison.

## Team Members and Roles

| Team Member | Responsibility |
|---|---|
| Tyson Seran | Data collection and cleaning; model, repo setup and merging everything |
| Luis Caushaj | Charts |
| Jeanne Laignel | Streamlit interface and application |
| Kristian Hekimov | README documentation and model testing |
| Emmanuel Tokei | Slides, AI log and presenting |



## Limitations

- **Barcelona only:** The model is trained on Barcelona Airbnb data and should not be applied to other cities without retraining.
- **Simple linear model:** Linear regression cannot capture every complex relationship influencing Airbnb prices.
- **Changing market prices:** Nightly rates fluctuate because of seasonality, demand, special events, and changes in supply.
- **Prediction uncertainty:** The model's reported average error of approximately €81 per night limits the precision of its recommendations.
- **Unobserved factors:** Factors such as interior quality, views, and special amenities may not be fully reflected in the model.

## Data Source

[Inside Airbnb](https://insideairbnb.com/) — publicly available Airbnb listing data used for research and analysis.

## Disclaimer

Rip-off Radar BCN provides estimated prices for informational purposes. Its classifications are not guarantees that a property is underpriced or overpriced.