import requests
import json
import numpy as np

# Generate synthetic load data (approx 40k to 60k)
loads = [50000 + 1000 * np.sin(i) for i in range(24)]
payload = {"last_24_hours": loads}

print("🚀 Sending request to Render backend...")
try:
    response = requests.post("https://loadiq.onrender.com/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        pred = data.get("predicted_load", 0)
        
        print("\n📊 RESPONSE DATA:")
        print(json.dumps(data, indent=2))
        
        print("\n🧐 ANALYSIS:")
        if pred < 100:
            print("❌ FAILURE: Predicted Load is too small (~0-100).")
            print("   -> The server is likely using an uninitialized or identity scaler.")
            print("   -> ACTION: You must trigger a new deployment with the correct scaler file.")
        else:
            print(f"✅ SUCCESS: Predicted Load is realistic ({pred:,.0f} MW).")
            print("   -> The server has the updated scaler.")
            
    else:
        print("❌ Error Response:")
        print(response.text)
except Exception as e:
    print(f"❌ Connection Failed: {e}")
