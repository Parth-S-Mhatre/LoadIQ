
import requests
import json
import numpy as np

def test_model2():
    url = "http://127.0.0.1:8000/predict"
    # 24 hours of dummy data
    payload = {
        "last_24_hours": [2000.0 + i*10 for i in range(24)]
    }
    try:
        resp = requests.post(url, json=payload)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_model2()
