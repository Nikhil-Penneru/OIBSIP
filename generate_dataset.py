"""
Generates a realistic synthetic retail sales dataset (~10,000 transactions).
Includes seasonal patterns, age-category correlations, and regional variation.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

N_TRANSACTIONS = 10_000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)
DATE_RANGE_DAYS = (END_DATE - START_DATE).days

# ---------------------------------------------------------------------------
# Product catalogue
# ---------------------------------------------------------------------------
PRODUCTS = {
    'Electronics': {
        'items': ['Smartphone', 'Laptop', 'Tablet', 'Headphones', 'Smart Watch',
                  'Bluetooth Speaker', 'Power Bank', 'USB Cable'],
        'price_range': (15, 800),
    },
    'Clothing': {
        'items': ['T-Shirt', 'Jeans', 'Jacket', 'Sneakers', 'Dress',
                  'Hoodie', 'Shorts', 'Formal Shirt'],
        'price_range': (10, 150),
    },
    'Beauty': {
        'items': ['Moisturizer', 'Lipstick', 'Perfume', 'Face Wash',
                  'Sunscreen', 'Hair Oil', 'Foundation', 'Eye Shadow'],
        'price_range': (5, 120),
    },
    'Sports': {
        'items': ['Yoga Mat', 'Dumbbells', 'Running Shoes', 'Water Bottle',
                  'Resistance Band', 'Skipping Rope', 'Cricket Bat', 'Football'],
        'price_range': (8, 200),
    },
    'Food': {
        'items': ['Organic Honey', 'Protein Bar', 'Green Tea', 'Dark Chocolate',
                  'Mixed Nuts', 'Olive Oil', 'Peanut Butter', 'Granola'],
        'price_range': (3, 40),
    },
    'Home & Kitchen': {
        'items': ['Blender', 'Cookware Set', 'Bedsheet', 'Pillow',
                  'Table Lamp', 'Wall Clock', 'Knife Set', 'Storage Container'],
        'price_range': (10, 250),
    },
}

CATEGORIES = list(PRODUCTS.keys())

# ---------------------------------------------------------------------------
# Seasonal weighting  (holiday spike in Nov-Dec, slow Jan-Feb)
# ---------------------------------------------------------------------------
MONTH_WEIGHTS = {
    1: 0.60, 2: 0.65, 3: 0.80, 4: 0.85, 5: 0.90, 6: 1.00,
    7: 1.00, 8: 0.95, 9: 1.00, 10: 1.10, 11: 1.40, 12: 1.50,
}

all_days = [START_DATE + timedelta(days=d) for d in range(DATE_RANGE_DAYS + 1)]
day_weights = np.array([MONTH_WEIGHTS[d.month] for d in all_days], dtype=float)
day_weights /= day_weights.sum()

date_indices = np.random.choice(len(all_days), size=N_TRANSACTIONS, p=day_weights)
dates = [all_days[i] for i in date_indices]

# ---------------------------------------------------------------------------
# Age → Category affinity
# ---------------------------------------------------------------------------
AGE_CATEGORY_PROBS = {
    'young':  [0.30, 0.25, 0.15, 0.15, 0.05, 0.10],   # 18-30
    'middle': [0.20, 0.15, 0.10, 0.10, 0.15, 0.30],   # 31-50
    'older':  [0.10, 0.10, 0.20, 0.05, 0.25, 0.30],   # 51-70
}

# ---------------------------------------------------------------------------
# Customer base
# ---------------------------------------------------------------------------
N_CUSTOMERS = 2000
customer_ids = [f'CUST_{str(i).zfill(5)}' for i in range(1, N_CUSTOMERS + 1)]
customer_meta = {
    cid: {
        'gender': np.random.choice(['Male', 'Female'], p=[0.48, 0.52]),
        'age': int(np.random.randint(18, 71)),
        'region': np.random.choice(['North', 'South', 'East', 'West']),
    }
    for cid in customer_ids
}

PAYMENT_METHODS = ['Cash', 'Credit Card', 'Debit Card', 'UPI']
PAYMENT_PROBS = [0.20, 0.30, 0.25, 0.25]

QUANTITY_VALUES = list(range(1, 8))
QUANTITY_PROBS = [0.30, 0.25, 0.20, 0.12, 0.07, 0.04, 0.02]

# ---------------------------------------------------------------------------
# Build rows
# ---------------------------------------------------------------------------
rows = []
for i in range(N_TRANSACTIONS):
    cid = np.random.choice(customer_ids)
    meta = customer_meta[cid]
    age = meta['age']

    age_group = 'young' if age <= 30 else ('middle' if age <= 50 else 'older')
    cat = np.random.choice(CATEGORIES, p=AGE_CATEGORY_PROBS[age_group])

    prod_info = PRODUCTS[cat]
    product = np.random.choice(prod_info['items'])
    price = round(np.random.uniform(*prod_info['price_range']), 2)
    qty = int(np.random.choice(QUANTITY_VALUES, p=QUANTITY_PROBS))

    rows.append({
        'Transaction ID': 1001 + i,
        'Date': dates[i].strftime('%Y-%m-%d'),
        'Customer ID': cid,
        'Gender': meta['gender'],
        'Age': age,
        'Product Category': cat,
        'Product': product,
        'Quantity': qty,
        'Price per Unit': price,
        'Total Amount': round(price * qty, 2),
        'Region': meta['region'],
        'Payment Method': np.random.choice(PAYMENT_METHODS, p=PAYMENT_PROBS),
    })

df = pd.DataFrame(rows).sort_values('Date').reset_index(drop=True)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'retail_sales_data.csv')
df.to_csv(out_path, index=False)

print(f"[OK] Dataset saved -> {out_path}")
print(f"     Shape         : {df.shape}")
print(f"     Date range    : {df['Date'].min()} -> {df['Date'].max()}")
print(f"     Columns       : {list(df.columns)}")
