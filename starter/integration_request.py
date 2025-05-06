import json
import requests

data = {
    "age": 50,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 83311,
    "education": "HS-grad",
    "education_num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Craft-repair",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native-country": "United-States",
}

response = requests.post(
    "https://fabio-ellena-predict-salary.onrender.com/predict", data=json.dumps(data)
)

print(response.status_code)
print(response.json())
