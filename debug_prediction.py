import requests
import json
import numpy as np

# Generate synthetic load data (approx 40k to 60k)
loads = [50000 + 1000 * np.sin(i) for i in range(24)]
payload = {"last_24_hours": loads}

try:
    response = requests.post("http://localhost:8000/debug_predict", json=payload)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    else:
        print("Error Response:")
        print(response.text)
except Exception as e:
    print(f"Request Failed: {e}")
