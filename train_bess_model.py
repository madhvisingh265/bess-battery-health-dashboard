import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1. LOAD DATASET
# =========================

DATA_PATH = "metadata.xlsx"

df = pd.read_excel(DATA_PATH)

print("Dataset Loaded Successfully")
print(df.head())

# =========================
# 2. CLEAN COLUMN NAMES
# =========================

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("\nColumns:")
print(df.columns)

# =========================
# 3. CREATE CYCLE NUMBER
# =========================

df = df.reset_index(drop=True)
df["cycle_number"] = np.arange(1, len(df) + 1)

# =========================
# 4. FIND CAPACITY COLUMN
# =========================

capacity_col = None

for col in df.columns:
    if "capacity" in col:
        capacity_col = col
        break

if capacity_col is None:
    raise Exception("Capacity column not found in dataset")

print(f"\nUsing capacity column: {capacity_col}")

df = df.dropna(subset=[capacity_col])
df[capacity_col] = pd.to_numeric(df[capacity_col], errors='coerce')
df = df.dropna(subset=[capacity_col])
# =========================
# 5. CALCULATE SOH
# =========================

initial_capacity = df[capacity_col].iloc[0]

df["soh"] = (df[capacity_col] / initial_capacity) * 100
df["capacity_fade"] = 100 - df["soh"]

# =========================
# 6. RUL CALCULATION
# =========================

eol_cycles = df[df["soh"] <= 70]["cycle_number"]

if len(eol_cycles) > 0:
    eol_cycle = eol_cycles.iloc[0]
else:
    eol_cycle = df["cycle_number"].max()

df["rul_cycles"] = eol_cycle - df["cycle_number"]
df["rul_cycles"] = df["rul_cycles"].apply(lambda x: max(x, 0))

# =========================
# 7. HEALTH STATUS
# =========================

def health_status(soh):
    if soh >= 85:
        return "Healthy"
    elif soh >= 70:
        return "Warning"
    elif soh >= 60:
        return "Critical"
    else:
        return "Emergency"

df["health_status"] = df["soh"].apply(health_status)

# =========================
# 8. FEATURES
# =========================

possible_features = [
    "cycle_number",
    "ambient_temperature",
    "re",
    "rct",
    "voltage",
    "current",
    "temperature"
]

features = [col for col in possible_features if col in df.columns]

if len(features) == 0:
    features = ["cycle_number"]

print("\nFeatures used:", features)

X = df[features]
y = df["soh"]

X = X.fillna(X.mean(numeric_only=True))

# =========================
# 9. TRAIN MODEL
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# =========================
# 10. EVALUATE
# =========================

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

# =========================
# 11. SAVE MODEL
# =========================

os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

joblib.dump(model, "model/bess_model.pkl")
joblib.dump(features, "model/features.pkl")

df.to_csv("outputs/processed_data.csv", index=False)

print("\nModel saved successfully!")