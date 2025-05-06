import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from starter.ml.model import (
    train_model,
    compute_model_metrics,
    inference,
)


@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    return X, y


@pytest.fixture
def sample_dataframe():
    """Fixture to provide a sample dataframe for testing evaluate_slices."""
    data = {
        "feature1": ["A", "A", "B", "B", "C"],
        "feature2": ["X", "Y", "X", "Y", "X"],
        "label": [0, 1, 0, 1, 0],
    }
    return pd.DataFrame(data)


def test_train_model(sample_data):
    """Test the train_model function."""
    X, y = sample_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics():
    """Test the compute_model_metrics function."""
    y = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    print(f"Precision: {precision}, Recall: {recall}, F-beta: {fbeta}")

    assert precision is not None
    assert recall is not None
    assert fbeta is not None


def test_inference(sample_data):
    """Test the inference function."""
    X, y = sample_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})
