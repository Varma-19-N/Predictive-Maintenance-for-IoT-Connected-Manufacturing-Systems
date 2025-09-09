# save_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
def main():
    print("🚀 Loading dataset...")
    df = pd.read_csv('data/iot_sensor_data_8500.csv')
    
    # Feature Engineering
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
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Scale features (important for XGBoost consistency)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost (your best model for recall)
    xgb_balanced = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb_balanced.fit(X_train_scaled, y_train_balanced)
    
    # Save model and scaler
    joblib.dump(xgb_balanced, 'models/xgb_failure_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(features, 'models/feature_names.pkl')
    
    print("✅ Model, scaler, and features saved to 'models/' folder!")

if __name__ == "__main__":
    main()
