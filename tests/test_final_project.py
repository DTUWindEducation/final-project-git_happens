import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from __init__ import WindFarmDataset, Evaluation, Prediction, WindFarmPlotter  # __init__ here refers to src/__init__.py

#from src import WindFarmDataset, Evaluation, Prediction  

@pytest.fixture
def sample_data():
    date_rng = pd.date_range(start='2023-01-01', periods=5, freq='h')
    data = pd.DataFrame({
        'Power': [10, 20, 30, 40, 50],
        'windspeed_100m': [3.1, 3.5, 4.0, 4.2, 4.5]
    }, index=date_rng)
    return data

def test_invalid_site_index():
    invalid_index = 9
    assert invalid_index not in [1, 2, 3, 4], "Should be a valid site index"

def test_summary_without_data():
    ds = WindFarmDataset("dummy.csv")
    with pytest.raises(AttributeError):
        ds.summary()

def test_create_lagged_features(sample_data):
    dataset = WindFarmDataset(file_path='fake.csv')
    dataset.data = sample_data.copy()
    lagged_data = dataset.create_lagged_features()
    assert 'Power_lag_1' in lagged_data.columns
    assert 'windspeed_100m_lag_1' in lagged_data.columns
    assert len(lagged_data) == 4
    assert lagged_data['Power_lag_1'].iloc[0] == 10  # First value should be the first Power value
    assert lagged_data['windspeed_100m_lag_1'].iloc[0] == 3.1  # First value should be the first windspeed value

    

def test_split_data(sample_data):
    dataset = WindFarmDataset(file_path='fake.csv')
    dataset.data = sample_data.copy()
    X_train, X_test, y_train, y_test = dataset.split_data()
    assert len(X_train) == 3
    assert len(X_test) == 1
    assert 'Power_lag_1' in X_train.columns
    assert 'Power_lag_1' in X_test.columns
    assert 'windspeed_100m_lag_1' in X_train.columns
    assert 'windspeed_100m_lag_1' in X_test.columns
    assert len(y_train) == 3
    assert len(y_test) == 1


def test_compute_metrics_zero_error():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4])
    eval = Evaluation(y_true, y_pred)
    mse, mae, rmse = eval.compute_metrics()
    assert mse == 0
    assert mae == 0
    assert rmse == 0

def test_compute_metrics_different_lengths():
    y_true = np.array([10, 20, 30, 40])  # length 4
    y_pred = np.array([25, 35, 45])      # length 3
    
    eval = Evaluation(y_true, y_pred)
    eval.compute_metrics()
    
    # After shift, y_true should be [20, 30, 40]
    assert np.array_equal(eval.y_true, np.array([20, 30, 40]))

@pytest.fixture
def prediction_data():
    X_train = pd.DataFrame({'a_lag_1': [1, 2, 3], 'b_lag_1': [4, 5, 6]})
    y_train = pd.Series([10, 15, 20])
    X_test = pd.DataFrame({'a_lag_1': [3, 4], 'b_lag_1': [6, 7]})
    y_test = pd.Series([20, 25])
    return X_test, y_test, X_train, y_train

def test_linear_regression_prediction(prediction_data):
    X_test, y_test, X_train, y_train = prediction_data
    pred = Prediction(X_test, y_test, X_train, y_train)
    y_pred = pred.train_linear_regression()
    assert len(y_pred) == len(X_test)


def test_svm_prediction(prediction_data):
    X_test, y_test, X_train, y_train = prediction_data
    pred = Prediction(X_test, y_test, X_train, y_train)
    y_pred = pred.train_svm()
    assert len(y_pred) == len(X_test)

def test_persistence_model(prediction_data):
    X_test, y_test, X_train, y_train = prediction_data
    pred = Prediction(X_test, y_test, X_train, y_train)
    y_pred = pred.persistence_model()
    assert len(y_pred) == len(y_test) - 1
    assert y_pred[1] == y_test.iloc[0]  # Second prediction should be the first actual value

@pytest.mark.parametrize("a,b,expected_mse,expected_mae,expected_rmse", [
([1,2,3], [1,2,3], 0, 0, 0),  # perfect prediction
([1,2,3], [2,2,2], 0.67, 0.67, 0.82),  # rough RMSE
])

def test_rmse_values(a, b, expected_mse, expected_mae, expected_rmse):
    """
    Test RMSE values for different inputs.
    """
    ev = Evaluation(a, b)
    mse, mae, rmse = ev.compute_metrics()
    assert round(rmse, 2) == expected_rmse
    assert round(mse, 2) == expected_mse
    assert round(mae, 2) == expected_mae

def test_plot_data_runs_without_error():

    # Create dummy time-series data
    date_range = pd.date_range(start="2023-01-01", periods=10, freq="h")
    data = pd.DataFrame({"site1": np.random.rand(10)}, index=date_range)

    plotter = WindFarmPlotter(data["site1"])

    try:
        plotter.plot_data(
            site_index=1,
            start_time="2023-01-01 00:00",
            end_time="2023-01-01 09:00",
            title="Test Plot"
        )
        
    except Exception as e:
        pytest.fail(f"Plotting failed with error: {e}")
    