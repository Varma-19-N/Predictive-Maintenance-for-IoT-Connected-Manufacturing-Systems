# phase2_imbalance_fix.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE  # ← Magic tool to balance classes
import plotly.express as px
import plotly.graph_objects as go

print("🚀 Loading dataset...")
df = pd.read_csv('data/iot_sensor_data_8500.csv')

# =============================================
# 🧠 FEATURE ENGINEERING (Same as Phase 1)
# =============================================
df['sensor_01_rolling_avg'] = df['sensor_01'].rolling(window=10, min_periods=1).mean()
df['sensor_03_spike'] = (df['sensor_03'] > df['sensor_03'].rolling(window=10, min_periods=1).mean() * 2).astype(int)
df['sensor_05_std_last_5'] = df['sensor_05'].rolling(window=5, min_periods=1).std().fillna(0)

features = [
    'sensor_01', 'sensor_02', 'sensor_03', 'sensor_04', 'sensor_05', 'sensor_06',
    'sensor_01_rolling_avg', 'sensor_03_spike', 'sensor_05_std_last_5'
]
X = df[features]
y = df['failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =============================================
# ⚖️ APPLY SMOTE TO BALANCE TRAINING DATA
# =============================================
print("⚖️  Applying SMOTE to fix class imbalance...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"After SMOTE:  {pd.Series(y_train_balanced).value_counts().to_dict()}")

# =============================================
# 🌲 TRAIN RANDOM FOREST (on balanced data)
# =============================================
print("\n🌳 Training Random Forest on balanced data...")
rf_balanced = RandomForestClassifier(n_estimators=100, random_state=42)
rf_balanced.fit(X_train_balanced, y_train_balanced)

y_pred_rf_bal = rf_balanced.predict(X_test)
acc_rf_bal = accuracy_score(y_test, y_pred_rf_bal)
print(f"✅ Balanced RF Accuracy: {acc_rf_bal:.2%}")

# =============================================
# ⚡ TRAIN XGBOOST (on balanced data)
# =============================================
print("\n⚡ Training XGBoost on balanced data...")
xgb_balanced = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb_balanced.fit(X_train_balanced, y_train_balanced)

y_pred_xgb_bal = xgb_balanced.predict(X_test)
acc_xgb_bal = accuracy_score(y_test, y_pred_xgb_bal)
print(f"✅ Balanced XGBoost Accuracy: {acc_xgb_bal:.2%}")

# =============================================
# 📊 SHOW DETAILED REPORT (Focus on RECALL for Failures!)
# =============================================
print("\n📋 Classification Report - Balanced Random Forest (Focus on Class 1 RECALL):")
print(classification_report(y_test, y_pred_rf_bal))

print("\n📋 Classification Report - Balanced XGBoost (Focus on Class 1 RECALL):")
print(classification_report(y_test, y_pred_xgb_bal))

# =============================================
# 📈 PLOT CONFUSION MATRIX (Visualize Improvement)
# =============================================
fig = go.Figure()

# Add before/after comparison (optional later) — for now, just show current
cm = confusion_matrix(y_test, y_pred_rf_bal)
fig.add_trace(go.Heatmap(
    z=cm,
    x=['Predicted Normal', 'Predicted Failure'],
    y=['Actual Normal', 'Actual Failure'],
    text=cm, texttemplate="%{text}", colorscale='Blues',
    showscale=False
))

fig.update_layout(title="Confusion Matrix - Balanced Random Forest", width=500, height=400)
fig.write_image("confusion_matrix_rf_balanced.png")  # Saves image
fig.show()  # Opens in browser if in notebook, else just saves

print("\n📊 Confusion matrix saved as 'confusion_matrix_rf_balanced.png'")

# =============================================
# 🏆 DECLARE WINNER (Now based on RECALL, not just accuracy)
# =============================================
report_rf = classification_report(y_test, y_pred_rf_bal, output_dict=True)
report_xgb = classification_report(y_test, y_pred_xgb_bal, output_dict=True)

recall_rf_failure = report_rf['1']['recall']
recall_xgb_failure = report_xgb['1']['recall']

print(f"\n🎯 RECALL for Failures (What matters most!):")
print(f"   Random Forest: {recall_rf_failure:.2%}")
print(f"   XGBoost:       {recall_xgb_failure:.2%}")

if recall_rf_failure > recall_xgb_failure:
    print(f"\n🏆 Random Forest wins for FAILURE RECALL! ({recall_rf_failure:.2%})")
    BEST_MODEL = rf_balanced
else:
    print(f"\n🏆 XGBoost wins for FAILURE RECALL! ({recall_xgb_failure:.2%})")
    BEST_MODEL = xgb_balanced

print("\n🎉 PHASE 2 (Part 1) COMPLETE — CLASS IMBALANCE FIXED!")
print("➡️  Next: Building Interactive Dashboard with Streamlit")