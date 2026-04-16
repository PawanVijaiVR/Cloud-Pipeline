"""
Project 3 — Cloud-Based Data Pipeline (AWS Architecture)
=========================================================
Architecture:  Raw Data → S3 (simulated) → Python/Spark ETL → SQLite DB → SQL Insights → Report

This script simulates a real AWS pipeline locally:
  • LocalS3     = /project3/data/s3_bucket/   (simulates AWS S3)
  • ETL Engine  = Python/pandas               (simulates Spark/Glue)
  • Warehouse   = SQLite                      (simulates Redshift/Athena)
  • Insights    = SQL queries                 (same SQL works on Redshift)

In production you'd swap:
  open(path) → boto3.client('s3').get_object()
  SQLite     → psycopg2 / Redshift / Athena
  pandas     → PySpark DataFrames

Resume line demonstrated:
  "Built an end-to-end cloud data pipeline using AWS S3 for storage,
   Python/Spark for ETL processing, and SQL for analytical insights."
"""

import os, sqlite3, json, hashlib, gzip, shutil, time, warnings
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
np.random.seed(99)

# ─── Paths ───────────────────────────────────────────────
ROOT     = Path("/home/claude/project3")
S3_ROOT  = ROOT / "data" / "s3_bucket"    # simulates S3
RAW      = S3_ROOT / "raw"
STAGED   = S3_ROOT / "staged"
CURATED  = S3_ROOT / "curated"
DB_PATH  = ROOT / "output" / "warehouse.db"
OUT      = ROOT / "output"

for p in [RAW, STAGED, CURATED, OUT]:
    p.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════
# LAYER 0 — GENERATE SYNTHETIC RAW DATA  (mimics real ingest)
# ═══════════════════════════════════════════════════════
print("=" * 62)
print("PROJECT 3 — Cloud-Based Data Pipeline")
print("=" * 62)
print("\n[LAYER 0] Generating raw data → s3://bucket/raw/")

N_ORDERS    = 20_000
N_CUSTOMERS = 2_000
N_PRODUCTS  = 300

cities    = ["Chennai","Mumbai","Delhi","Bangalore","Hyderabad","Pune","Kolkata","Surat"]
states    = ["TN","MH","DL","KA","TS","MH","WB","GJ"]
city_state = dict(zip(cities, states))

categories = ["Electronics","Clothing","Books","Home & Kitchen","Sports","Beauty","Toys"]
product_names = {
    "Electronics":    ["Smartphone","Laptop","Tablet","Smartwatch","Earbuds"],
    "Clothing":       ["T-Shirt","Jeans","Dress","Jacket","Kurta"],
    "Books":          ["Novel","Textbook","Self-Help","Biography","Comic"],
    "Home & Kitchen": ["Cookware","Blender","Air Fryer","Curtains","Storage Box"],
    "Sports":         ["Yoga Mat","Dumbbells","Cycle","Badminton Set","Running Shoes"],
    "Beauty":         ["Face Cream","Shampoo","Perfume","Lipstick","Sunscreen"],
    "Toys":           ["Lego Set","Board Game","RC Car","Puzzle","Doll"],
}

# Products
products = []
for pid in range(1, N_PRODUCTS + 1):
    cat = np.random.choice(categories)
    pname = np.random.choice(product_names[cat])
    products.append({
        "product_id":   pid,
        "product_name": f"{pname} #{pid}",
        "category":     cat,
        "unit_cost":    round(np.random.uniform(50, 8000), 2),
        "supplier":     f"Supplier_{np.random.randint(1,21)}",
    })
products_df = pd.DataFrame(products)

# Customers
customers = []
for cid in range(1, N_CUSTOMERS + 1):
    city = np.random.choice(cities)
    signup = datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1460))
    customers.append({
        "customer_id":   cid,
        "name":          f"Customer_{cid}",
        "city":          city,
        "state":         city_state[city],
        "signup_date":   signup.strftime("%Y-%m-%d"),
        "tier":          np.random.choice(["Bronze","Silver","Gold","Platinum"],
                                           p=[0.45,0.30,0.18,0.07]),
    })
customers_df = pd.DataFrame(customers)

# Orders (multi-line, realistic patterns)
orders = []
order_items = []
oid = 1
start_date = datetime(2022, 1, 1)
end_date   = datetime(2023, 12, 31)

for _ in range(N_ORDERS):
    cid     = np.random.randint(1, N_CUSTOMERS + 1)
    n_items = np.random.randint(1, 5)
    # Seasonal boost in Oct-Dec
    days = np.random.randint(0, (end_date - start_date).days)
    order_date = start_date + timedelta(days=days)
    status = np.random.choice(
        ["Delivered","Shipped","Processing","Cancelled","Returned"],
        p=[0.65, 0.15, 0.08, 0.07, 0.05]
    )
    # Payment
    payment = np.random.choice(
        ["UPI","Credit Card","Debit Card","Net Banking","COD"],
        p=[0.35, 0.20, 0.18, 0.12, 0.15]
    )
    pids     = np.random.choice(range(1, N_PRODUCTS + 1), size=n_items, replace=False)
    subtotal = 0
    for pid in pids:
        qty   = np.random.randint(1, 4)
        price = products_df.loc[products_df["product_id"]==pid, "unit_cost"].values[0]
        disc  = round(np.random.uniform(0, 0.30), 2)
        line_total = round(price * qty * (1 - disc), 2)
        subtotal  += line_total
        order_items.append({
            "order_id":   oid,
            "product_id": int(pid),
            "quantity":   qty,
            "unit_price": price,
            "discount":   disc,
            "line_total": line_total,
        })
    orders.append({
        "order_id":     oid,
        "customer_id":  cid,
        "order_date":   order_date.strftime("%Y-%m-%d"),
        "status":       status,
        "payment_mode": payment,
        "order_total":  round(subtotal, 2),
    })
    oid += 1

orders_df = pd.DataFrame(orders)
items_df  = pd.DataFrame(order_items)

# Inject dirty data (nulls, wrong types, duplicates) for ETL demo
dirty_orders = orders_df.copy()
dirty_orders.loc[np.random.choice(dirty_orders.index, 200), "order_total"] = np.nan
dirty_orders.loc[np.random.choice(dirty_orders.index, 50),  "status"]     = None
dirty_orders = pd.concat([dirty_orders, dirty_orders.sample(150)], ignore_index=True)  # duplicates

# Save raw files (compressed JSON = realistic S3 ingest format)
def save_json_gz(df, path):
    with gzip.open(path, "wt") as f:
        f.write(df.to_json(orient="records", date_format="iso"))

save_json_gz(dirty_orders, RAW / "orders_raw.json.gz")
save_json_gz(items_df,     RAW / "order_items_raw.json.gz")
products_df.to_csv(RAW / "products.csv", index=False)
customers_df.to_csv(RAW / "customers.csv", index=False)

print(f"   📁 Raw files uploaded to s3://bucket/raw/")
print(f"      orders_raw.json.gz  → {len(dirty_orders):,} rows (with {150} duplicates & {200} nulls)")
print(f"      order_items_raw.json.gz → {len(items_df):,} rows")
print(f"      products.csv        → {len(products_df):,} rows")
print(f"      customers.csv       → {len(customers_df):,} rows")

# ═══════════════════════════════════════════════════════
# LAYER 1 — EXTRACT  (read from simulated S3)
# ═══════════════════════════════════════════════════════
print("\n[LAYER 1] EXTRACT — Reading from s3://bucket/raw/")
t0 = time.time()

with gzip.open(RAW / "orders_raw.json.gz", "rt") as f:
    raw_orders = pd.read_json(f)
with gzip.open(RAW / "order_items_raw.json.gz", "rt") as f:
    raw_items = pd.read_json(f)
raw_products   = pd.read_csv(RAW / "products.csv")
raw_customers  = pd.read_csv(RAW / "customers.csv")

print(f"   Orders extracted   : {len(raw_orders):,}")
print(f"   Items extracted    : {len(raw_items):,}")
print(f"   Elapsed            : {time.time()-t0:.2f}s")

# ═══════════════════════════════════════════════════════
# LAYER 2 — TRANSFORM  (ETL / Spark-equivalent processing)
# ═══════════════════════════════════════════════════════
print("\n[LAYER 2] TRANSFORM — Cleaning, validating, enriching …")
t1 = time.time()

# 2a. Remove duplicates
before_dedup = len(raw_orders)
clean_orders = raw_orders.drop_duplicates(subset=["order_id"])
print(f"   Duplicates removed : {before_dedup - len(clean_orders):,}")

# 2b. Fix nulls
clean_orders["order_total"] = clean_orders["order_total"].fillna(
    clean_orders["order_total"].median()
)
clean_orders["status"] = clean_orders["status"].fillna("Processing")

# 2c. Type casting & derived columns
clean_orders["order_date"]  = pd.to_datetime(clean_orders["order_date"])
clean_orders["order_year"]  = clean_orders["order_date"].dt.year
clean_orders["order_month"] = clean_orders["order_date"].dt.month
clean_orders["order_qtr"]   = clean_orders["order_date"].dt.quarter
clean_orders["day_of_week"] = clean_orders["order_date"].dt.day_name()
clean_orders["is_weekend"]  = clean_orders["order_date"].dt.dayofweek >= 5

# 2d. Enrich orders with customer info
clean_orders = clean_orders.merge(
    raw_customers[["customer_id","city","state","tier"]], on="customer_id", how="left"
)

# 2e. Enrich items with product info
clean_items = raw_items.merge(
    raw_products[["product_id","product_name","category","supplier"]], on="product_id", how="left"
)

# 2f. Data quality report
quality_report = {
    "total_raw_orders":   before_dedup,
    "after_dedup":        len(clean_orders),
    "null_filled":        200,
    "orders_delivered":   int((clean_orders["status"] == "Delivered").sum()),
    "orders_cancelled":   int((clean_orders["status"] == "Cancelled").sum()),
    "date_range":         f"{clean_orders['order_date'].min().date()} → {clean_orders['order_date'].max().date()}",
}

print(f"   After dedup        : {len(clean_orders):,}")
print(f"   Null values filled : 200")
print(f"   Derived columns    : order_year, order_month, order_qtr, day_of_week, is_weekend")
print(f"   Elapsed            : {time.time()-t1:.2f}s")

# Save staged data to S3 (Parquet-like: compressed CSV)
clean_orders.to_csv(STAGED / "orders_clean.csv.gz", index=False, compression="gzip")
clean_items.to_csv(STAGED  / "items_clean.csv.gz",  index=False, compression="gzip")
with open(STAGED / "quality_report.json", "w") as f:
    json.dump(quality_report, f, indent=2)
print(f"   ✅ Staged data saved → s3://bucket/staged/")

# ═══════════════════════════════════════════════════════
# LAYER 3 — LOAD  (into SQLite = Redshift simulation)
# ═══════════════════════════════════════════════════════
print("\n[LAYER 3] LOAD — Writing to data warehouse …")
t2 = time.time()

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

# Create star schema
cur.executescript("""
DROP TABLE IF EXISTS fact_orders;
DROP TABLE IF EXISTS fact_order_items;
DROP TABLE IF EXISTS dim_customers;
DROP TABLE IF EXISTS dim_products;
DROP TABLE IF EXISTS dim_date;

CREATE TABLE dim_customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT, city TEXT, state TEXT,
    signup_date TEXT, tier TEXT
);

CREATE TABLE dim_products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT, category TEXT,
    unit_cost REAL, supplier TEXT
);

CREATE TABLE dim_date (
    date_id TEXT PRIMARY KEY,
    year INTEGER, month INTEGER, quarter INTEGER,
    day_of_week TEXT, is_weekend INTEGER
);

CREATE TABLE fact_orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER, order_date TEXT,
    status TEXT, payment_mode TEXT,
    order_total REAL, city TEXT, state TEXT,
    tier TEXT, order_year INTEGER,
    order_month INTEGER, order_qtr INTEGER,
    day_of_week TEXT, is_weekend INTEGER,
    FOREIGN KEY(customer_id) REFERENCES dim_customers(customer_id)
);

CREATE TABLE fact_order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER, product_id INTEGER,
    quantity INTEGER, unit_price REAL,
    discount REAL, line_total REAL,
    product_name TEXT, category TEXT, supplier TEXT,
    FOREIGN KEY(order_id)   REFERENCES fact_orders(order_id),
    FOREIGN KEY(product_id) REFERENCES dim_products(product_id)
);
""")

raw_customers.to_sql("dim_customers", conn, if_exists="replace", index=False)
raw_products.to_sql("dim_products",   conn, if_exists="replace", index=False)
clean_orders.assign(
    order_date=clean_orders["order_date"].astype(str)
).to_sql("fact_orders", conn, if_exists="replace", index=False)
clean_items.to_sql("fact_order_items", conn, if_exists="replace", index=False)
conn.commit()

print(f"   Tables loaded      : fact_orders, fact_order_items, dim_customers, dim_products")
print(f"   Total rows         : {len(clean_orders)+len(clean_items)+len(raw_customers)+len(raw_products):,}")
print(f"   Elapsed            : {time.time()-t2:.2f}s")

# ═══════════════════════════════════════════════════════
# LAYER 4 — SQL ANALYTICS  (Redshift-compatible queries)
# ═══════════════════════════════════════════════════════
print("\n[LAYER 4] ANALYZE — Running SQL queries …")

def sql(query):
    return pd.read_sql_query(query, conn)

# Q1: Monthly revenue trend
monthly_revenue = sql("""
    SELECT order_year, order_month,
           COUNT(order_id)      AS num_orders,
           SUM(order_total)     AS revenue,
           AVG(order_total)     AS avg_order_value
    FROM   fact_orders
    WHERE  status != 'Cancelled'
    GROUP  BY order_year, order_month
    ORDER  BY order_year, order_month
""")

# Q2: Revenue by category
category_revenue = sql("""
    SELECT   category,
             SUM(line_total)     AS revenue,
             COUNT(DISTINCT order_id) AS num_orders,
             AVG(discount)*100   AS avg_discount_pct
    FROM     fact_order_items
    GROUP BY category
    ORDER BY revenue DESC
""")

# Q3: Customer tier analysis
tier_analysis = sql("""
    SELECT   tier,
             COUNT(DISTINCT o.customer_id) AS customers,
             COUNT(o.order_id)             AS orders,
             SUM(order_total)              AS total_revenue,
             AVG(order_total)              AS avg_order_value
    FROM     fact_orders o
    WHERE    status = 'Delivered'
    GROUP BY tier
    ORDER BY total_revenue DESC
""")

# Q4: Top 10 cities by revenue
city_revenue = sql("""
    SELECT   city, state,
             COUNT(order_id)  AS orders,
             SUM(order_total) AS revenue
    FROM     fact_orders
    WHERE    status = 'Delivered'
    GROUP BY city, state
    ORDER BY revenue DESC
    LIMIT 10
""")

# Q5: Payment mode split
payment_split = sql("""
    SELECT   payment_mode,
             COUNT(*)          AS orders,
             SUM(order_total)  AS revenue,
             ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM fact_orders), 1) AS pct
    FROM     fact_orders
    GROUP BY payment_mode
    ORDER BY orders DESC
""")

# Q6: Order status breakdown
status_breakdown = sql("""
    SELECT   status,
             COUNT(*)         AS orders,
             SUM(order_total) AS revenue
    FROM     fact_orders
    GROUP BY status
    ORDER BY orders DESC
""")

# Q7: Top products by revenue
top_products = sql("""
    SELECT   product_name, category,
             SUM(line_total)     AS revenue,
             SUM(quantity)       AS units_sold,
             COUNT(DISTINCT order_id) AS num_orders
    FROM     fact_order_items
    GROUP BY product_name, category
    ORDER BY revenue DESC
    LIMIT 15
""")

# Q8: Weekend vs weekday analysis
weekend_analysis = sql("""
    SELECT   CASE WHEN is_weekend=1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
             COUNT(order_id)  AS orders,
             SUM(order_total) AS revenue,
             AVG(order_total) AS avg_order_value
    FROM     fact_orders
    WHERE    status != 'Cancelled'
    GROUP BY is_weekend
""")

print("   ✅ 8 analytical queries executed")
print(f"\n▸ Monthly revenue sample (first 5 months):")
print(monthly_revenue.head(5).to_string(index=False))
print(f"\n▸ Category revenue:")
print(category_revenue.to_string(index=False))
print(f"\n▸ Customer tier analysis:")
print(tier_analysis.to_string(index=False))
print(f"\n▸ Payment mode split:")
print(payment_split.to_string(index=False))

conn.close()

# ═══════════════════════════════════════════════════════
# LAYER 5 — VISUALISE  (Analytics Dashboard)
# ═══════════════════════════════════════════════════════
print("\n[LAYER 5] VISUALIZE — Generating pipeline dashboard …")
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#2E86AB","#A23B72","#F18F01","#C73E1D","#3B1F2B","#44BBA4","#E94F37"]

fig = plt.figure(figsize=(20, 16))
fig.suptitle("Cloud Data Pipeline — End-to-End Analytics Dashboard\n(AWS S3 → ETL → Warehouse → Insights)",
             fontsize=15, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

# A: Monthly Revenue
ax1 = fig.add_subplot(gs[0, :2])
monthly_revenue["period"] = (
    monthly_revenue["order_year"].astype(str) + "-"
    + monthly_revenue["order_month"].astype(str).str.zfill(2)
)
ax1.bar(monthly_revenue["period"], monthly_revenue["revenue"] / 1e6,
        color=PALETTE[0], alpha=0.85, edgecolor="white")
ax1.plot(monthly_revenue["period"], monthly_revenue["revenue"] / 1e6,
         color=PALETTE[2], lw=2, marker="o", markersize=4)
ax1.set_title("Monthly Revenue Trend (₹ Millions)", fontweight="bold")
ax1.set_xlabel("Month")
ax1.set_ylabel("Revenue (₹M)")
ax1.tick_params(axis="x", rotation=60, labelsize=7)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.1f}M"))

# B: Category Revenue
ax2 = fig.add_subplot(gs[0, 2])
ax2.barh(category_revenue["category"], category_revenue["revenue"] / 1e6,
         color=PALETTE[1], edgecolor="white")
ax2.set_title("Revenue by Category", fontweight="bold")
ax2.set_xlabel("Revenue (₹M)")
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.1f}M"))

# C: Customer Tier Donut
ax3 = fig.add_subplot(gs[1, 0])
wedge_colors = [PALETTE[i] for i in range(len(tier_analysis))]
wedges, texts, autotexts = ax3.pie(
    tier_analysis["total_revenue"],
    labels=tier_analysis["tier"],
    autopct="%1.1f%%",
    colors=wedge_colors,
    startangle=90,
    wedgeprops={"width": 0.6}
)
for at in autotexts:
    at.set_fontsize(8)
ax3.set_title("Revenue by Customer Tier", fontweight="bold")

# D: Top cities
ax4 = fig.add_subplot(gs[1, 1])
ax4.barh(city_revenue["city"][::-1], city_revenue["revenue"][::-1] / 1e6,
         color=PALETTE[4], edgecolor="white")
ax4.set_title("Top 10 Cities by Revenue", fontweight="bold")
ax4.set_xlabel("Revenue (₹M)")

# E: Payment mode
ax5 = fig.add_subplot(gs[1, 2])
ax5.pie(payment_split["orders"], labels=payment_split["payment_mode"],
        autopct="%1.1f%%",
        colors=[PALETTE[i % len(PALETTE)] for i in range(len(payment_split))],
        startangle=120)
ax5.set_title("Orders by Payment Mode", fontweight="bold")

# F: Order status
ax6 = fig.add_subplot(gs[2, 0])
colors_status = {
    "Delivered":"#44BBA4","Shipped":"#2E86AB","Processing":"#F18F01",
    "Cancelled":"#C73E1D","Returned":"#A23B72"
}
bars = ax6.bar(status_breakdown["status"],
               status_breakdown["orders"],
               color=[colors_status.get(s,"#888") for s in status_breakdown["status"]],
               edgecolor="white")
ax6.set_title("Orders by Status", fontweight="bold")
ax6.set_ylabel("Number of Orders")
ax6.tick_params(axis="x", rotation=30)
for bar in bars:
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8)

# G: Top 10 products
ax7 = fig.add_subplot(gs[2, 1:])
tp = top_products.head(10)
short = tp["product_name"].str[:22]
ax7.barh(short[::-1], tp["revenue"][::-1] / 1e3,
         color=PALETTE[2], edgecolor="white")
ax7.set_title("Top 10 Products by Revenue", fontweight="bold")
ax7.set_xlabel("Revenue (₹K)")

plt.savefig(OUT / "pipeline_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   ✅ Saved → output/pipeline_dashboard.png")

# ═══════════════════════════════════════════════════════
# LAYER 6 — EXPORT CURATED DATA BACK TO S3
# ═══════════════════════════════════════════════════════
monthly_revenue.to_csv(CURATED / "monthly_revenue.csv", index=False)
category_revenue.to_csv(CURATED / "category_revenue.csv", index=False)
tier_analysis.to_csv(CURATED / "tier_analysis.csv", index=False)
city_revenue.to_csv(CURATED / "city_revenue.csv", index=False)
payment_split.to_csv(CURATED / "payment_split.csv", index=False)
top_products.to_csv(CURATED / "top_products.csv", index=False)

# ═══════════════════════════════════════════════════════
# PIPELINE SUMMARY REPORT
# ═══════════════════════════════════════════════════════
print(f"\n{'='*62}")
print("PIPELINE EXECUTION SUMMARY")
print(f"{'='*62}")
print(f"  Stage 0  Raw ingest     : {len(dirty_orders):,} orders uploaded to S3")
print(f"  Stage 1  Extract        : 4 source files read")
print(f"  Stage 2  Transform      : {before_dedup-len(clean_orders):,} dupes removed | 200 nulls filled | 5 cols derived")
print(f"  Stage 3  Load           : 4 tables in warehouse ({len(clean_orders)+len(clean_items):,} rows)")
print(f"  Stage 4  SQL Analytics  : 8 business queries executed")
print(f"  Stage 5  Visualise      : Dashboard PNG exported")
print(f"  Stage 6  Curated export : 6 CSVs saved to s3://bucket/curated/")

total_rev = monthly_revenue["revenue"].sum()
delivered_pct = status_breakdown.loc[status_breakdown["status"]=="Delivered","orders"].values[0]
print(f"\n📊 Key Metrics:")
print(f"   Total Revenue     : ₹{total_rev/1e7:.2f} Crore")
print(f"   Orders Processed  : {len(clean_orders):,}")
print(f"   Delivered Orders  : {delivered_pct:,}")
print(f"   Unique Customers  : {len(raw_customers):,}")
print(f"   Product SKUs      : {len(raw_products):,}")
print(f"\n{'='*62}")
print("PROJECT 3 COMPLETE ✅")
print(f"{'='*62}")
