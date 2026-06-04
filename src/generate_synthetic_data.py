import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Generate 1000 product sales records
n = 1000
product_ids = np.random.choice([f"P{str(i).zfill(3)}" for i in range(1, 21)], n)
dates = [datetime(2023,1,1) + timedelta(days=int(np.random.uniform(0, 365))) for _ in range(n)]
units_sold = np.random.poisson(lam=5, size=n)
price = np.random.uniform(10, 100, n)
cost_price = price * np.random.uniform(0.4, 0.8, n)
discount_applied = np.random.uniform(0, 0.3, n)
current_stock = np.random.randint(0, 200, n)

df = pd.DataFrame({
    'product_id': product_ids,
    'date': dates,
    'units_sold': units_sold,
    'price': price,
    'cost_price': cost_price,
    'discount_applied': discount_applied,
    'current_stock': current_stock
})

df.to_csv('data/synthetic/sales_data.csv', index=False)
print("Synthetic data saved to data/synthetic/sales_data.csv")