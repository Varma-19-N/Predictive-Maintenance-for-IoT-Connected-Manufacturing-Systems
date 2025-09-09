# phase1_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

print("🚀 Loading dataset...")
df = pd.read_csv('data/iot_sensor_data_8500.csv')
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# =============================================
# 🧠 FEATURE ENGINEERING (Secret Weapon)
# =============================================
print("🔧 Engineering features...")

# Rolling average of vibration (sensor_01) — smooths out noise
df['sensor_01_rolling_avg'] = df['sensor_01'].rolling(window=10, min_periods=1).mean()

# Spike detector: is current pressure 2x higher than recent average?
df['sensor_03_spike'] = (
    df['sensor_03'] > df['sensor_03'].rolling(window=10, min_periods=1).mean() * 2
).astype(int)

# Volatility: standard deviation of voltage over last 5 readings
df['sensor_05_std_last_5'] = df['sensor_05'].rolling(window=5, min_periods=1).std().fillna(0)

print("✅ 3 new features created!")

# =============================================
# 🎯 PREPARE DATA FOR TRAINING
# =============================================
features = [
    'sensor_01', 'sensor_02', 'sensor_03', 'sensor_04', 'sensor_05', 'sensor_06',
    'sensor_01_rolling_avg', 'sensor_03_spike', 'sensor_05_std_last_5'
]

X = df[features]
y = df['failure']

# Split data — stratify to keep failure ratio same in train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Training on {X_train.shape[0]} samples, Testing on {X_test.shape[0]}")

# =============================================
# 🌲 TRAIN RANDOM FOREST
# =============================================
print("\n🌳 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"✅ Random Forest Accuracy: {acc_rf:.2%}")

# =============================================
# ⚡ TRAIN XGBOOST
# =============================================
print("\n⚡ Training XGBoost...")
xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"✅ XGBoost Accuracy: {acc_xgb:.2%}")

# =============================================
# 📊 SHOW DETAILED REPORT
# =============================================
print("\n📋 Classification Report - Random Forest:")
print(classification_report(y_test, y_pred_rf))

print("\n📋 Classification Report - XGBoost:")
print(classification_report(y_test, y_pred_xgb))

# =============================================
# 🏆 COMPARE & DECLARE WINNER
# =============================================
if acc_rf > acc_xgb:
    print(f"\n🏆 Random Forest wins! ({acc_rf:.2%} vs {acc_xgb:.2%})")
else:
    print(f"\n🏆 XGBoost wins! ({acc_xgb:.2%} vs {acc_rf:.2%})")

print("\n🎉 PHASE 1 COMPLETE!")