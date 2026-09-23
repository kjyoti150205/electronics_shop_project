"""
train_model.py
Electronic Shop Sales Prediction — ML Training Pipeline
Target: revenue = product_price * number_of_ordered_items
Uses RandomForestRegressor for better capture of product-price-driven non-linearity.
"""

import os, json, datetime, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "Electronics_Shop_Dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
IMG_DIR    = os.path.join(BASE_DIR, "report_images")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

print("=" * 60)
print("  Electronic Shop Sales Prediction — Training Pipeline")
print("=" * 60)

# ── 1. Load & engineer ────────────────────────────────────────────────────────
print("\n[1/6] Loading dataset …")
BASE_DATE = datetime.date(1899, 12, 30)
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(
    df["date"].apply(lambda x: BASE_DATE + datetime.timedelta(days=int(x))))
df["revenue"] = df["product_price"] * df["number_of_ordered_items"]
df = df[df["order_status"].isin(["Delivered", "In Process"])].copy()
df["day_of_week"] = df["date"].dt.dayofweek
df["month"]       = df["date"].dt.month
df["state"]       = df["location"].astype(str).str.strip()
print(f"      Rows after filter: {len(df):,}  |  Revenue mean=${df['revenue'].mean():.2f}")

# ── 2. Encode ─────────────────────────────────────────────────────────────────
print("\n[2/6] Encoding categoricals …")
le_product = LabelEncoder()
le_channel  = LabelEncoder()
le_state    = LabelEncoder()
df["product_enc"] = le_product.fit_transform(df["product_name"])
df["channel_enc"] = le_channel.fit_transform(df["channel_of_ordering"])
df["state_enc"]   = le_state.fit_transform(df["state"])
joblib.dump(le_product, os.path.join(MODEL_DIR, "le_product.pkl"))
joblib.dump(le_channel,  os.path.join(MODEL_DIR, "le_channel.pkl"))
joblib.dump(le_state,    os.path.join(MODEL_DIR, "le_state.pkl"))

# Add product_price as a feature (strong driver, available at order time)
FEATURES = ["product_enc", "channel_enc", "state_enc",
            "day_of_week", "month", "product_price", "number_of_ordered_items"]
TARGET   = "revenue"

# ── 3. Split ──────────────────────────────────────────────────────────────────
print("\n[3/6] Splitting 80/20 (random_state=42) …")
X = df[FEATURES]; y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"      Train {len(X_train):,} | Test {len(X_test):,}")

# ── 4. Scale & train ──────────────────────────────────────────────────────────
print("\n[4/6] Training RandomForestRegressor (n_estimators=200) …")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train_s, y_train)
joblib.dump(model,  os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
print("\n[5/6] Evaluating …")
y_pred = model.predict(X_test_s)
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"      R²={r2:.4f}  MAE=${mae:.2f}  RMSE=${rmse:.2f}")

metrics = {"r2": round(r2,4), "mae": round(mae,2), "rmse": round(rmse,2)}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

importances = dict(zip(FEATURES, model.feature_importances_.tolist()))
model_info = {
    "intercept": 0,
    "coefficients": {k: round(v, 4) for k, v in importances.items()},
    "feature_names": FEATURES,
    "product_classes": le_product.classes_.tolist(),
    "channel_classes": le_channel.classes_.tolist(),
    "state_classes":   le_state.classes_.tolist(),
}
with open(os.path.join(MODEL_DIR, "model_info.json"), "w") as f:
    json.dump(model_info, f, indent=2)

# ── 6. Plots ──────────────────────────────────────────────────────────────────
print("\n[6/6] Generating plots …")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Model Diagnostics — Electronic Shop Sales Prediction", fontsize=13)
axes[0].scatter(y_test, y_pred, alpha=0.3, color="#2563eb", s=10)
lim = [0, max(y_test.max(), y_pred.max())]
axes[0].plot(lim, lim, "r--", lw=1.5)
axes[0].set_xlabel("Actual Revenue ($)"); axes[0].set_ylabel("Predicted Revenue ($)")
axes[0].set_title(f"Actual vs Predicted  (R²={r2:.3f})"); axes[0].grid(alpha=0.3)
residuals = y_test - y_pred
axes[1].hist(residuals, bins=50, color="#16a34a", edgecolor="white", alpha=0.8)
axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
axes[1].set_xlabel("Residual ($)"); axes[1].set_ylabel("Frequency")
axes[1].set_title("Residual Distribution"); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "diagnostics.png"), dpi=150)
plt.close()

# EDA charts
rev_prod = df.groupby("product_name")["revenue"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 5))
rev_prod.plot(kind="bar", ax=ax, color="#3b82f6", edgecolor="white")
ax.set_title("Average Revenue by Product", fontsize=13); ax.set_xlabel("")
ax.set_ylabel("Avg Revenue ($)"); ax.tick_params(axis="x", rotation=45); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "revenue_by_product.png"), dpi=150); plt.close()

rev_ch = df.groupby("channel_of_ordering")["revenue"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4))
rev_ch.plot(kind="bar", ax=ax, color="#8b5cf6", edgecolor="white")
ax.set_title("Average Revenue by Channel", fontsize=13); ax.set_xlabel("")
ax.set_ylabel("Avg Revenue ($)"); ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "revenue_by_channel.png"), dpi=150); plt.close()

monthly = df.groupby(df["date"].dt.to_period("M"))["revenue"].sum()
fig, ax = plt.subplots(figsize=(10, 4))
monthly.plot(ax=ax, color="#f59e0b", marker="o", linewidth=2)
ax.set_title("Monthly Total Revenue Trend", fontsize=13)
ax.set_xlabel("Month"); ax.set_ylabel("Total Revenue ($)"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "monthly_revenue.png"), dpi=150); plt.close()

feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
feat_imp.plot(kind="barh", ax=ax, color="#3b82f6", edgecolor="white")
ax.set_title("Feature Importances (RandomForest)", fontsize=13)
ax.set_xlabel("Importance"); ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "feature_coefficients.png"), dpi=150); plt.close()

print("\n" + "=" * 60)
print(f"  Done!  R²={r2:.4f}  MAE=${mae:.2f}  RMSE=${rmse:.2f}")
print("=" * 60)
