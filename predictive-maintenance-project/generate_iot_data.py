import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_records = 8500
start_time = datetime(2024, 1, 1, 0, 0, 0)
machines = [f"M{str(i).zfill(3)}" for i in range(1, 11)]  # M001 to M010

# Generate timestamps (every 5 minutes)
timestamps = [start_time + timedelta(minutes=5*i) for i in range(n_records)]

# Assign machines in round-robin (so each machine gets ~850 records)
machine_ids = [machines[i % len(machines)] for i in range(n_records)]

# Generate base sensor readings with small random walk (realistic drift)
sensor_01 = np.cumsum(np.random.normal(0, 0.02, n_records)) + 0.5  # vibration
sensor_02 = np.cumsum(np.random.normal(0, 0.1, n_records)) + 32.0  # temperature
sensor_03 = np.cumsum(np.random.normal(0, 0.05, n_records)) + 1.0   # pressure
sensor_04 = np.cumsum(np.random.normal(0, 0.03, n_records)) + 4.5   # current
sensor_05 = np.cumsum(np.random.normal(0, 0.5, n_records)) + 220.0  # voltage
sensor_06 = np.cumsum(np.random.normal(0, 5, n_records)) + 1800     # rpm

# Clip to realistic ranges
sensor_01 = np.clip(sensor_01, 0.1, 3.0)
sensor_02 = np.clip(sensor_02, 25.0, 60.0)
sensor_03 = np.clip(sensor_03, 0.5, 15.0)
sensor_04 = np.clip(sensor_04, 3.0, 15.0)
sensor_05 = np.clip(sensor_05, 200.0, 280.0)
sensor_06 = np.clip(sensor_06, 1500, 2500)

# CREATE FAILURE LABELS based on realistic conditions:
# If vibration > 1.8 AND temp > 45 AND pressure > 8.0 → FAILURE (1)
failure = (
    (sensor_01 > 1.8) &
    (sensor_02 > 45.0) &
    (sensor_03 > 8.0)
).astype(int)

# Force at least 12% failure rate (realistic for predictive maintenance)
current_failure_rate = failure.sum() / len(failure)
if current_failure_rate < 0.12:
    # Randomly add failures to reach ~12-15%
    n_needed = int(0.12 * n_records) - failure.sum()
    zero_indices = np.where(failure == 0)[0]
    add_failures_at = np.random.choice(zero_indices, size=n_needed, replace=False)
    failure[add_failures_at] = 1

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'machine_id': machine_ids,
    'sensor_01': sensor_01,
    'sensor_02': sensor_02,
    'sensor_03': sensor_03,
    'sensor_04': sensor_04,
    'sensor_05': sensor_05,
    'sensor_06': sensor_06,
    'failure': failure
})

# Save to CSV
df.to_csv('data/iot_sensor_data_8500.csv', index=False)

print("✅ Dataset generated successfully!")
print(f"📁 Saved to: data/iot_sensor_data_8500.csv")
print(f"📊 Shape: {df.shape}")
print(f"⚠️ Failure Rate: {df['failure'].mean():.2%}")
print(df.head())