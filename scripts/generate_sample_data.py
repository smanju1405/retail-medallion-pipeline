"""Generate synthetic, deliberately messy Retail Co raw CSVs for the medallion pipeline.

All names, emails and amounts are invented. The messiness is intentional so the
profiler and the cleansing rules have something to find:
  - mixed date formats, currency symbols, blanks and text numbers
  - negative quantities, exact duplicate rows
  - region/category casing and whitespace variants
  - some product_ids that do not exist in products (orphans)

Usage:  python scripts/generate_sample_data.py
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "sample_data"
SEED = 42
REGIONS = ["North", "South", "East", "West", "Central"]
CATEGORIES = ["Grocery", "Apparel", "Electronics", "Home", "Beauty"]
FIRST = ["Asha", "Ben", "Chen", "Dara", "Elif", "Farid", "Grace", "Hiro", "Ines", "Jamal"]
LAST = ["Patel", "Okafor", "Lindqvist", "Garcia", "Kim", "Haddad", "Novak", "Silva", "Tan", "Moreau"]


def _messy_date(d: date, rng: random.Random) -> str:
    style = rng.random()
    if style < 0.6:
        return d.isoformat()                      # 2026-03-05
    if style < 0.8:
        return d.strftime("%d-%b-%Y")             # 05-Mar-2026
    return d.strftime("%m/%d/%Y")                 # 03/05/2026


def make_customers(rng: random.Random, n: int = 80) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        email = f"{name.lower().replace(' ', '.')}{i}@example.com"
        region = rng.choice(REGIONS)
        region = rng.choice([region, region.lower(), f" {region} ", region.upper()])
        signup = date(2024, 1, 1) + timedelta(days=rng.randint(0, 700))
        rows.append({"customer_id": f"C{i:04d}", "customer_name": name, "email": email,
                     "region": region, "signup_date": _messy_date(signup, rng)})
    rows.append(dict(rows[0]))  # exact duplicate row
    return pd.DataFrame(rows)


def make_products(rng: random.Random, n: int = 25) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        category = rng.choice(CATEGORIES)
        price = round(rng.uniform(3, 180), 2)
        rows.append({"product_id": f"P{i:03d}", "product_name": f"{category} item {i}",
                     "category": rng.choice([category, category.lower()]),
                     "list_price": f"${price:,.2f}" if rng.random() < 0.5 else f"{price:.2f}"})
    return pd.DataFrame(rows)


def make_sales(rng: random.Random, customers: int = 80, products: int = 25, n: int = 600) -> pd.DataFrame:
    rows = []
    start = date(2026, 1, 1)
    for i in range(1, n + 1):
        d = start + timedelta(days=rng.randint(0, 270))
        qty = rng.randint(1, 6)
        unit = round(rng.uniform(3, 180), 2)
        line_total = round(qty * unit, 2)
        qty_cell = str(qty)
        roll = rng.random()
        if roll < 0.03:
            qty_cell = ""                         # blank quantity
        elif roll < 0.05:
            qty_cell = str(-qty)                  # negative quantity (return keyed wrong)
        elif roll < 0.06:
            qty_cell = "two"                      # text number
        total_cell = f"${line_total:,.2f}" if rng.random() < 0.4 else f"{line_total:.2f}"
        if rng.random() < 0.04:
            total_cell = ""                       # missing line total
        region = rng.choice(REGIONS)
        region = rng.choice([region, region.lower(), f"{region} "])
        product = f"P{rng.randint(1, products + 2):03d}"  # +2 creates orphan product ids
        rows.append({
            "order_id": f"O{i:05d}",
            "order_date": _messy_date(d, rng),
            "customer_id": "" if rng.random() < 0.02 else f"C{rng.randint(1, customers):04d}",
            "product_id": product,
            "store_region": region,
            "quantity": qty_cell,
            "unit_price": f"{unit:.2f}",
            "line_total": total_cell,
        })
    rows.append(dict(rows[10]))  # exact duplicate sales line
    rows.append(dict(rows[200]))
    return pd.DataFrame(rows)


def main() -> None:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_customers(rng).to_csv(OUT_DIR / "customers_raw.csv", index=False)
    make_products(rng).to_csv(OUT_DIR / "products_raw.csv", index=False)
    make_sales(rng).to_csv(OUT_DIR / "sales_raw.csv", index=False)
    print(f"Wrote 3 synthetic raw CSVs to {OUT_DIR}")


if __name__ == "__main__":
    main()
