"""Generate a realistic 500-row e-commerce sales dataset for testing.

Run from the project root:
    python examples/generate_sample_data.py

Output: examples/sample_sales.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_ROWS = 500
OUTPUT_PATH = Path(__file__).parent / "sample_sales.csv"

CATEGORIES = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
REGIONS = ["North", "South", "East", "West", "Central"]
SEGMENTS = ["Premium", "Standard", "Budget"]

PRODUCTS: dict[str, list[str]] = {
    "Electronics": [
        "Laptop Pro 15", "Wireless Earbuds", "4K Monitor", "USB-C Hub",
        "Mechanical Keyboard", "Webcam HD", "Smart Speaker", "Phone Stand",
        "SSD External 1TB", "LED Desk Lamp",
    ],
    "Clothing": [
        "Merino Wool Sweater", "Performance Tee", "Chino Pants", "Running Shoes",
        "Winter Jacket", "Canvas Backpack", "Compression Socks", "Baseball Cap",
        "Fleece Hoodie", "Slim Fit Jeans",
    ],
    "Home & Garden": [
        "French Press Coffee Maker", "Air Purifier", "Bamboo Cutting Board",
        "Scented Candle Set", "Throw Pillow Cover", "Wall Clock Modern",
        "Plant Pot Ceramic", "Kitchen Scale Digital", "Vacuum Storage Bags",
        "LED String Lights",
    ],
    "Sports": [
        "Yoga Mat Premium", "Resistance Bands Set", "Water Bottle 32oz",
        "Foam Roller", "Jump Rope Speed", "Gym Gloves", "Protein Shaker",
        "Ankle Weights", "Pull-up Bar", "Workout Timer",
    ],
    "Books": [
        "Python for Data Analysis", "SQL Cookbook", "The Pragmatic Programmer",
        "Thinking in Systems", "Atomic Habits", "Deep Work", "The Phoenix Project",
        "Designing Data-Intensive Applications", "Clean Code", "The Data Warehouse Toolkit",
    ],
}

# Realistic base prices per category (log-normal-ish)
PRICE_PARAMS: dict[str, tuple[float, float]] = {
    "Electronics": (4.5, 0.7),    # mean ~$90, right-skewed, goes up to ~$500
    "Clothing": (3.8, 0.5),       # mean ~$45
    "Home & Garden": (3.6, 0.6),  # mean ~$37
    "Sports": (3.2, 0.5),         # mean ~$24
    "Books": (2.9, 0.3),          # mean ~$18
}


def generate() -> pd.DataFrame:
    """Generate the sample sales dataset with realistic distributions and flaws."""
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------ #
    # Order IDs and dates
    # ------------------------------------------------------------------ #
    order_ids = [f"ORD-{i:05d}" for i in range(1, N_ROWS + 1)]
    # Skew dates toward more recent — simulate business growth over a year
    date_weights = np.linspace(0.5, 1.5, 365)
    date_weights /= date_weights.sum()
    date_pool = pd.date_range("2023-01-01", periods=365, freq="D")
    date_indices = rng.choice(len(date_pool), size=N_ROWS, p=date_weights)
    dates = date_pool[date_indices]

    # ------------------------------------------------------------------ #
    # Customers — ~150 unique customers with repeat purchases
    # ------------------------------------------------------------------ #
    n_customers = 150
    customer_pool = [f"CUST-{i:04d}" for i in range(1, n_customers + 1)]
    # Power-law: top 20% of customers do ~50% of purchases
    customer_weights = np.array([1 / (i + 1) ** 0.6 for i in range(n_customers)])
    customer_weights /= customer_weights.sum()
    customer_ids = rng.choice(customer_pool, size=N_ROWS, p=customer_weights)

    # ------------------------------------------------------------------ #
    # Product categories and names
    # ------------------------------------------------------------------ #
    # Electronics slightly over-represented (reflects real portfolio skew)
    category_weights = [0.25, 0.22, 0.20, 0.18, 0.15]
    categories = rng.choice(CATEGORIES, size=N_ROWS, p=category_weights)
    product_names = [
        rng.choice(PRODUCTS[cat]) for cat in categories
    ]

    # ------------------------------------------------------------------ #
    # Prices — log-normal per category
    # ------------------------------------------------------------------ #
    unit_prices = np.array([
        float(np.exp(rng.normal(*PRICE_PARAMS[cat]))) for cat in categories
    ])
    unit_prices = np.round(unit_prices, 2)

    # ------------------------------------------------------------------ #
    # Quantities — mostly 1-5, with a few bulk outliers
    # ------------------------------------------------------------------ #
    quantities = rng.integers(1, 6, size=N_ROWS).astype(float)
    # Inject 3 bulk purchase outliers
    bulk_indices = rng.choice(N_ROWS, size=3, replace=False)
    quantities[bulk_indices] = rng.integers(50, 101, size=3).astype(float)

    # ------------------------------------------------------------------ #
    # Discounts — 60% of orders get no discount; 40% get 5–30%
    # ------------------------------------------------------------------ #
    discount_mask = rng.random(N_ROWS) > 0.6
    discount_pcts = np.zeros(N_ROWS)
    discount_pcts[discount_mask] = rng.uniform(5, 30, size=discount_mask.sum())
    discount_pcts = np.round(discount_pcts, 1)

    # ------------------------------------------------------------------ #
    # Revenue
    # ------------------------------------------------------------------ #
    total_revenue = np.round(
        quantities * unit_prices * (1 - discount_pcts / 100), 2
    )

    # ------------------------------------------------------------------ #
    # Regions and segments
    # ------------------------------------------------------------------ #
    region_weights = [0.22, 0.18, 0.20, 0.25, 0.15]
    regions = rng.choice(REGIONS, size=N_ROWS, p=region_weights)

    # Premium customers are more common in West and East
    segment_weights_map = {
        "West": [0.35, 0.45, 0.20],
        "East": [0.30, 0.48, 0.22],
        "North": [0.20, 0.50, 0.30],
        "South": [0.18, 0.50, 0.32],
        "Central": [0.15, 0.52, 0.33],
    }
    segments_col = np.array([
        rng.choice(SEGMENTS, p=segment_weights_map[region])
        for region in regions
    ])

    # ------------------------------------------------------------------ #
    # Returns — 5% overall; higher for high-discount orders
    # ------------------------------------------------------------------ #
    return_probs = np.where(discount_pcts > 20, 0.12, 0.04)
    is_returned = (rng.random(N_ROWS) < return_probs).astype(int)

    # ------------------------------------------------------------------ #
    # Introduce realistic missing values (~2.8% of key fields)
    # ------------------------------------------------------------------ #
    # Missing quantity values — simulate data pipeline issue in Q4
    q4_mask = pd.DatetimeIndex(dates).month >= 10
    q4_indices = np.where(q4_mask)[0]
    n_missing_qty = max(1, int(len(q4_indices) * 0.06))
    missing_qty_idx = rng.choice(q4_indices, size=n_missing_qty, replace=False)
    quantities[missing_qty_idx] = np.nan

    # Missing unit_price — random, small number
    n_missing_price = max(1, int(N_ROWS * 0.015))
    missing_price_idx = rng.choice(N_ROWS, size=n_missing_price, replace=False)
    unit_prices[missing_price_idx] = np.nan

    # Missing region — random
    n_missing_region = max(1, int(N_ROWS * 0.01))
    missing_region_idx = rng.choice(N_ROWS, size=n_missing_region, replace=False)
    regions_list = regions.tolist()
    for idx in missing_region_idx:
        regions_list[idx] = None

    # ------------------------------------------------------------------ #
    # Assemble DataFrame
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(
        {
            "order_id": order_ids,
            "date": pd.DatetimeIndex(dates).strftime("%Y-%m-%d"),
            "customer_id": customer_ids,
            "product_category": categories,
            "product_name": product_names,
            "quantity": quantities,
            "unit_price": unit_prices,
            "total_revenue": total_revenue,
            "region": regions_list,
            "customer_segment": segments_col,
            "discount_pct": discount_pcts,
            "is_returned": is_returned,
        }
    )

    # Sort by date for a natural order
    df = df.sort_values("date").reset_index(drop=True)
    return df


def main() -> None:
    """Entry point: generate and save sample_sales.csv."""
    print(f"Generating {N_ROWS}-row e-commerce sales dataset (seed={SEED})...")
    df = generate()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to: {OUTPUT_PATH}")

    # Quick summary
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"Total revenue: ${df['total_revenue'].sum():,.2f}")
    print(f"Missing values: {df.isnull().sum().sum()} total")
    print(f"Return rate: {df['is_returned'].mean():.1%}")
    print(f"Categories: {df['product_category'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
