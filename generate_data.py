"""
Project 2 - E-Commerce Recommendation System
Step 1: Generate realistic synthetic dataset
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ── Config ──────────────────────────────────────────────
N_USERS    = 500
N_PRODUCTS = 200
N_RATINGS  = 15000

# ── Product catalogue ────────────────────────────────────
categories = {
    "Electronics":   ["Wireless Earbuds", "USB-C Hub", "Mechanical Keyboard", "Gaming Mouse",
                       "Webcam 1080p", "Portable SSD", "Smart Watch", "Phone Stand",
                       "LED Desk Lamp", "Noise Cancelling Headphones"],
    "Books":         ["Python Crash Course", "Clean Code", "Data Science Handbook",
                       "Atomic Habits", "Deep Work", "The Pragmatic Programmer",
                       "Design Patterns", "System Design Interview", "Zero to One", "Lean Startup"],
    "Clothing":      ["Cotton T-Shirt", "Slim Fit Jeans", "Running Shorts", "Hoodie",
                       "Polo Shirt", "Chino Pants", "Athletic Socks", "Baseball Cap",
                       "Leather Belt", "Formal Blazer"],
    "Home & Kitchen":["Air Fryer", "Coffee Maker", "Blender Pro", "Instant Pot",
                       "Non-stick Pan", "Knife Set", "Food Storage Containers",
                       "Electric Kettle", "Toaster Oven", "Rice Cooker"],
    "Sports":        ["Yoga Mat", "Resistance Bands", "Dumbbell Set", "Foam Roller",
                       "Jump Rope", "Water Bottle", "Gym Bag", "Running Shoes",
                       "Cycling Gloves", "Protein Shaker"],
}

products = []
pid = 1
for cat, items in categories.items():
    for item in items:
        base_price = np.random.choice([9.99, 14.99, 19.99, 24.99, 29.99,
                                       39.99, 49.99, 79.99, 99.99, 149.99])
        products.append({
            "product_id":   pid,
            "product_name": item,
            "category":     cat,
            "price":        base_price,
            "avg_rating":   round(np.random.uniform(3.2, 5.0), 1),
            "num_reviews":  np.random.randint(50, 5000),
        })
        pid += 1

# Pad to N_PRODUCTS with auto-generated names
extra_names = [f"Product {i}" for i in range(pid, N_PRODUCTS + 1)]
for i, name in enumerate(extra_names, start=pid):
    cat = np.random.choice(list(categories.keys()))
    products.append({
        "product_id":   i,
        "product_name": name,
        "category":     cat,
        "price":        round(np.random.uniform(5, 200), 2),
        "avg_rating":   round(np.random.uniform(2.5, 5.0), 1),
        "num_reviews":  np.random.randint(10, 2000),
    })

products_df = pd.DataFrame(products)

# ── Users ────────────────────────────────────────────────
first_names = ["Arjun","Priya","Ravi","Sneha","Kiran","Meera","Vikram","Ananya",
                "Rahul","Divya","Arun","Pooja","Sanjay","Nisha","Amit","Kavitha"]
last_names  = ["Kumar","Sharma","Patel","Singh","Reddy","Nair","Gupta","Mishra",
                "Joshi","Mehta","Iyer","Agarwal","Rao","Das","Verma","Pillai"]
cities      = ["Chennai","Mumbai","Delhi","Bangalore","Hyderabad","Pune","Kolkata","Ahmedabad"]

users = []
for uid in range(1, N_USERS + 1):
    age_group = np.random.choice(["18-25","26-35","36-45","46-60"], p=[0.25,0.40,0.25,0.10])
    users.append({
        "user_id":    uid,
        "user_name":  f"{np.random.choice(first_names)} {np.random.choice(last_names)}",
        "city":       np.random.choice(cities),
        "age_group":  age_group,
        "member_since": np.random.randint(2018, 2024),
    })
users_df = pd.DataFrame(users)

# ── Ratings matrix (simulate real purchase + rating behaviour) ───────────────
# Each user has preferred categories → higher rating probability
user_preferences = {}
for uid in range(1, N_USERS + 1):
    pref_cats = np.random.choice(list(categories.keys()),
                                  size=np.random.randint(1, 4), replace=False)
    user_preferences[uid] = list(pref_cats)

records = []
product_ids  = products_df["product_id"].values
product_cats = dict(zip(products_df["product_id"], products_df["category"]))

seen = set()
attempts = 0
while len(records) < N_RATINGS and attempts < N_RATINGS * 10:
    attempts += 1
    uid  = np.random.randint(1, N_USERS + 1)
    pid  = int(np.random.choice(product_ids))
    key  = (uid, pid)
    if key in seen:
        continue
    seen.add(key)

    pcat = product_cats[pid]
    if pcat in user_preferences.get(uid, []):
        rating = np.random.choice([4, 4, 5, 5, 5, 3], p=[0.2,0.2,0.2,0.2,0.15,0.05])
    else:
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05,0.10,0.35,0.35,0.15])

    records.append({
        "user_id":    uid,
        "product_id": pid,
        "rating":     rating,
        "timestamp":  pd.Timestamp("2022-01-01") + pd.Timedelta(days=np.random.randint(0, 730)),
    })

ratings_df = pd.DataFrame(records)

# ── Purchase history (binary — bought or not) ─────────────────────────────────
purchases = ratings_df[ratings_df["rating"] >= 4][["user_id","product_id","timestamp"]].copy()
purchases["quantity"] = np.random.randint(1, 4, size=len(purchases))
purchases["order_value"] = purchases["product_id"].map(
    dict(zip(products_df["product_id"], products_df["price"]))
) * purchases["quantity"]

# ── Save ─────────────────────────────────────────────────
out = "/home/claude/project2/data"
products_df.to_csv(f"{out}/products.csv", index=False)
users_df.to_csv(f"{out}/users.csv", index=False)
ratings_df.to_csv(f"{out}/ratings.csv", index=False)
purchases.to_csv(f"{out}/purchases.csv", index=False)

print("✅ Dataset generated")
print(f"   Products : {len(products_df):,}")
print(f"   Users    : {len(users_df):,}")
print(f"   Ratings  : {len(ratings_df):,}")
print(f"   Purchases: {len(purchases):,}")
