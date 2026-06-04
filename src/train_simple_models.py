import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score
import joblib
from pathlib import Path

# Load data
df = pd.read_csv('data/synthetic/sales_data.csv', parse_dates=['date'])

# Feature engineering (simplified)
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['profit_margin'] = (df['price'] - df['cost_price']) / df['price']

# Target: discount_applied (what we want to predict)
features = ['units_sold', 'current_stock', 'profit_margin', 'day_of_week', 'month', 'is_weekend']
X = df[features].fillna(0)
y_discount = df['discount_applied']

# Train Discount Predictor (Random Forest)
X_train, X_test, y_train, y_test = train_test_split(X, y_discount, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"Discount Predictor MAE: {mean_absolute_error(y_test, y_pred):.3f}")

# Train Slow Stock Classifier (binary target: 1 if units_sold < 2)
y_slow = (df['units_sold'] < 2).astype(int)
gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
gb.fit(X_train, y_slow[:len(X_train)])
y_pred_slow = gb.predict(X_test)
print(f"Slow Stock F1: {f1_score(y_slow[:len(X_test)], y_pred_slow):.3f}")

# Save models
Path('models').mkdir(exist_ok=True)
joblib.dump(rf, 'models/discount_rf.pkl')
joblib.dump(gb, 'models/slow_gb.pkl')
print("Models saved to models/")