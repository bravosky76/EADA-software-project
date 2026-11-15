import pandas as pd

# 1. Load the raw Inside Airbnb Barcelona data
df = pd.read_csv("data/listings.csv.gz")
print("Raw rows:", len(df))

# 2. Turn price text like "$1,200.00" into a number (removes any symbol, not just $)
df["price"] = df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
df["price"] = pd.to_numeric(df["price"], errors="coerce")
print("Rows with a price:", df["price"].notna().sum())

# 3. Remove missing and extreme prices
df = df[(df["price"] >= 10) & (df["price"] <= 1000)]
print("Rows after price filter:", len(df))

# 4. Get number of bathrooms from text like "1.5 baths"
df["bathrooms"] = df["bathrooms_text"].str.extract(r"(\d+\.?\d*)")[0].astype(float)
df.loc[df["bathrooms_text"].str.contains("half", case=False, na=False), "bathrooms"] = 0.5

# 5. Use the district as the neighbourhood (fall back to the smaller area if empty)
if df["neighbourhood_group_cleansed"].notna().sum() > 0:
    df["neighbourhood"] = df["neighbourhood_group_cleansed"]
else:
    df["neighbourhood"] = df["neighbourhood_cleansed"]

# 6. Keep only the columns we need
columns = ["price", "neighbourhood", "room_type", "accommodates", "bedrooms",
           "bathrooms", "beds", "review_scores_rating", "number_of_reviews",
           "latitude", "longitude"]
df = df[columns]

# Show how many values are missing in each column
print("\nMissing values per column:")
print(df.isna().sum())

# Drop rows with missing values
df = df.dropna()

# 7. Save the clean data
df.to_csv("data/clean_listings.csv", index=False)
print("\nClean rows:", len(df))