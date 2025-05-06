import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def client():
    # Enter the TestClient context, which will fire @startup:
    with TestClient(app) as c:
        yield c

# Mock data for testing
valid_input_data = {
    "age": 39,
    "workclass": "Private",
    "fnlgt": 77516,
    "education": "Doctorate",
    "education-num": 20,
    "marital-status": "Never-married",
    "occupation": "Exec-managerial",
    "relationship": "Unmarried",
    "race": "White",
    "sex": "Male",
    "capital-gain": 80000,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States",
}

# Mock data for testing the other possible output
alternative_input_data = {
    "age": 50,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 83311,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Craft-repair",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


def test_root_endpoint(client):
    """Test the GET method for the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Salary Prediction API!"}


def test_predict_salary_above_50k(client):
    """Test the POST /predict endpoint for a salary prediction >50K."""
    response = client.post("/predict", json=valid_input_data)
    assert response.status_code == 200
    assert response.json() == {"salary_prediction": ">50K"}


def test_predict_salary_below_50k(client):
    """Test the POST /predict endpoint for a salary prediction <=50K."""
    response = client.post("/predict", json=alternative_input_data)
    assert response.status_code == 200
    assert response.json() == {"salary_prediction": "<=50K"}
