import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load the clean data
df = pd.read_csv("data/clean_listings.csv")

# 2. Choose the inputs (X) and what we predict (y)
features = ["neighbourhood", "room_type", "accommodates", "bedrooms",
            "bathrooms", "beds", "review_scores_rating", "number_of_reviews"]
X = df[features]
y = df["price"]

# 3. Split into training data (80%) and test data (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Turn text columns into numbers (one-hot encoding)
preprocess = ColumnTransformer(
    [("text", OneHotEncoder(handle_unknown="ignore"), ["neighbourhood", "room_type"])],
    remainder="passthrough"
)

# 5. Build and train the linear regression model
model = Pipeline([("preprocess", preprocess), ("regressor", LinearRegression())])
model.fit(X_train, y_train)

# 6. Check how good it is on the test data
predictions = model.predict(X_test)
print("R2 score:", round(r2_score(y_test, predictions), 3))
print("Average error (euros):", round(mean_absolute_error(y_test, predictions), 2))

# 7. Save the model so the app can use it
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")