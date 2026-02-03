import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Create a scaler for Load (target)
# Assuming load is between 30,000 and 90,000 MW
# We add some buffer
data = np.array([0, 100000]).reshape(-1, 1)

scaler_y = MinMaxScaler()
scaler_y.fit(data)

print(f"New Scaler Y Min: {scaler_y.min_}")
print(f"New Scaler Y Scale: {scaler_y.scale_}")

joblib.dump(scaler_y, "Backend/scaler_y.save")
print("✅ Regenerated Backend/scaler_y.save")
