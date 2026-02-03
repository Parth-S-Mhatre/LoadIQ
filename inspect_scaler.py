import joblib
import os

try:
    scaler_y = joblib.load("Backend/scaler_y.save")
    print(f"Scaler type: {type(scaler_y)}")
    if hasattr(scaler_y, 'data_min_'):
        print(f"Data Min: {scaler_y.data_min_}")
    if hasattr(scaler_y, 'data_max_'):
        print(f"Data Max: {scaler_y.data_max_}")
    if hasattr(scaler_y, 'scale_'):
        print(f"Scale: {scaler_y.scale_}")
    if hasattr(scaler_y, 'min_'):
        print(f"Min: {scaler_y.min_}")
        
    # Test inverse transform
    test_val = [[0.5]]
    inv = scaler_y.inverse_transform(test_val)
    print(f"Inverse Transform of 0.5: {inv[0][0]}")

except Exception as e:
    print(f"Failed to load scaler: {e}")
