
import requests
import json

def test_model1():
    url = "http://127.0.0.1:8001/predict"
    payload = {
        "hour": 10,
        "day_of_week": 2,
        "month": 6,
        "temperature": 30.0 # extra
    }
    try:
        resp = requests.post(url, json=payload)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_model1()
