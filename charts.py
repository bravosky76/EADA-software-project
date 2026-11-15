import os

import matplotlib
matplotlib.use("Agg")  # save files without needing a screen
import matplotlib.pyplot as plt
import pandas as pd

# 1. Load the cleaned data and make sure the output folder exists
df = pd.read_csv("data/clean_listings.csv")
os.makedirs("images", exist_ok=True)
print("Rows loaded:", len(df))

BLUE = "#2a6fdb"

# 2. Chart 1: average price by neighbourhood (horizontal bar chart)
avg_by_area = df.groupby("neighbourhood")["price"].mean().sort_values()

plt.figure(figsize=(10, 6))
plt.barh(avg_by_area.index, avg_by_area.values, color=BLUE)
for i, value in enumerate(avg_by_area.values):
    plt.text(value + 3, i, f"€{value:.0f}", va="center")
plt.xlim(0, avg_by_area.max() * 1.12)  # leave room for the labels
plt.title("Average Airbnb price per night by neighbourhood (Barcelona)")
plt.xlabel("Average price per night (€)")
plt.ylabel("Neighbourhood")
plt.tight_layout()
plt.savefig("images/price_by_neighbourhood.png", dpi=150)
plt.close()

# 3. Chart 2: average price by room type (bar chart)
avg_by_room = df.groupby("room_type")["price"].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 6))
plt.bar(avg_by_room.index, avg_by_room.values, color=BLUE)
for i, value in enumerate(avg_by_room.values):
    plt.text(i, value + 3, f"€{value:.0f}", ha="center")
plt.title("Average Airbnb price per night by room type")
plt.xlabel("Room type")
plt.ylabel("Average price per night (€)")
plt.tight_layout()
plt.savefig("images/price_by_room_type.png", dpi=150)
plt.close()

# 4. Chart 3: map of listings, coloured by price
# Cap the colour scale at the 95th percentile so a few very expensive
# listings don't wash out the colours for everyone else.
plt.figure(figsize=(9, 8))
points = plt.scatter(df["longitude"], df["latitude"], c=df["price"],
                     cmap="viridis", s=5, alpha=0.6,
                     vmax=df["price"].quantile(0.95))
plt.colorbar(points, label="Price per night (€)")
plt.title("Map of Airbnb listings in Barcelona, coloured by price")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.gca().set_aspect(1 / 0.75)  # roughly correct east-west scale at Barcelona's latitude
plt.tight_layout()
plt.savefig("images/listings_map.png", dpi=150)
plt.close()

# 5. Chart 4: price vs number of guests (box plot)
guest_counts = sorted(df["accommodates"].unique())
prices_per_group = [df.loc[df["accommodates"] == n, "price"] for n in guest_counts]

plt.figure(figsize=(11, 6))
plt.boxplot(prices_per_group, showfliers=False)
plt.xticks(range(1, len(guest_counts) + 1), guest_counts)
plt.title("Price per night by number of guests the listing accommodates")
plt.xlabel("Number of guests (accommodates)")
plt.ylabel("Price per night (€)")
plt.tight_layout()
plt.savefig("images/price_by_accommodates.png", dpi=150)
plt.close()

print("Saved 4 charts in the images/ folder")
