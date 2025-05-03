import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pytest
import sys
import os
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from __init__ import WindFarmDataset, Evaluation, Prediction, WindFarmPlotter, get_valid_site_index, get_lag_hours  # __init__ here refers to src/__init__.py

#---- Tests for user inputs ----

def test_get_valid_site_index_valid_input():
    """Test with valid user input for site index."""
    # Simulate user input for site index
    with patch('builtins.input', return_value='2'):
        assert get_valid_site_index() == 2

def test_get_valid_site_index_invalid_then_valid_input(capsys):
    """Test with invalid input followed by valid input for site index."""
    # Simulate invalid input followed by valid input
    with patch('builtins.input', side_effect=['abc', '5', '3']):
        result = get_valid_site_index()
        captured = capsys.readouterr()

        assert result == 3
        assert "❌ Invalid input. Please enter a number." in captured.out
        assert "❌ Invalid input. Please enter a valid value for the location" in captured.out

def test_get_lag_hours_valid_1(capsys):
    """Test with valid user input for lag hours."""
    # Simulate user input for lag hours
    with patch("builtins.input", return_value="1"):
        result = get_lag_hours()
        captured = capsys.readouterr()
        assert result == 1
        assert "✅ You selected a forecasting horizon of 1 hour(s)." in captured.out


def test_get_lag_hours_valid_24(capsys):
    """Test with valid user input for lag hours."""
    # Simulate user input for lag hours
    with patch("builtins.input", return_value="24"):
        result = get_lag_hours()
        captured = capsys.readouterr()
        assert result == 24
        assert "✅ You selected a forecasting horizon of 24 hour(s)." in captured.out


def test_get_lag_hours_invalid_then_valid(capsys):
    """Test with invalid input followed by valid input for lag hours."""
    # Simulate invalid input followed by valid input
    with patch("builtins.input", side_effect=["abc", "5", "24"]):
        result = get_lag_hours()
        captured = capsys.readouterr()
        assert result == 24
        assert "❌ Invalid input. Please enter a valid number." in captured.out
        assert "❌ Invalid input. Please enter a number between 1 and 24." in captured.out
        assert "✅ You selected a forecasting horizon of 24 hour(s)." in captured.out

#---- Tests for WindFarmDataset split data function ----
@pytest.fixture
def sample_data():
    """Fixture to create sample data for testing."""
    # Create a sample DataFrame with datetime index and some dummy data
    date_rng = pd.date_range(start='2023-01-01', periods=5, freq='h')
    data = pd.DataFrame({
        'Power': [10, 20, 30, 40, 50],
        'windspeed_100m': [3.1, 3.5, 4.0, 4.2, 4.5]
    }, index=date_rng)
    return data

def test_split_data(sample_data,lag_hours=1,split_index=0.8):
    """Test the split_data method of WindFarmDataset."""
    # Create a sample DataFrame with datetime index and some dummy data
    dataset = WindFarmDataset(file_path='fake.csv')
    dataset.data = sample_data.copy()
    x_train, x_test, y_train, y_test = dataset.split_data(lag_hours=lag_hours,split_index=split_index)
    # when lag is 1, lenght of total data should be 4 (5-1) hence train=3 and test=1
    assert len(x_train) == 3
    assert len(x_test) == 1
    assert 'Power_lag_1' in x_train.columns
    assert 'Power_lag_1' in x_test.columns
    assert 'windspeed_100m_lag_1' in x_train.columns
    assert 'windspeed_100m_lag_1' in x_test.columns
    assert len(y_train) == 3
    assert len(y_test) == 1


#---- Tests for each prediction model----

@pytest.fixture
def prediction_data(sample_data,lag_hours=1,split_index=0.8):
    """Dummy prediction data for testing."""
    dataset = WindFarmDataset(file_path='fake.csv')
    dataset.data = sample_data.copy()
    x_train, x_test, y_train, y_test = dataset.split_data(lag_hours=lag_hours,split_index=split_index)
    return x_test, y_test, x_train, y_train

def test_persistence_model_output_length(prediction_data):
    """Test the output length of the persistence model."""
    x_test, y_test, x_train, y_train = prediction_data
    model = Prediction(x_test, y_test, x_train, y_train)
    y_pred = model.persistence_model(lag_hours=1)
    assert len(y_pred) == len(x_test), "Output length should match x_test length"

def test_linear_regression_prediction(prediction_data):
    """Test the output length of the linear regression model."""
    x_test, y_test, x_train, y_train = prediction_data
    pred = Prediction(x_test, y_test, x_train, y_train)
    y_pred = pred.train_linear_regression()
    assert len(y_pred) == len(x_test)

def test_svm_prediction(prediction_data):
    """Test the output length of the SVM model."""
    x_test, y_test, x_train, y_train = prediction_data
    pred = Prediction(x_test, y_test, x_train, y_train)
    y_pred = pred.train_svm()
    assert len(y_pred) == len(x_test)

def test_random_forest_prediction(prediction_data):
    """Test the output length of the Random Forest model."""
    x_test, y_test, x_train, y_train = prediction_data
    pred = Prediction(x_test, y_test, x_train, y_train)
    y_pred = pred.train_random_forest()
    assert len(y_pred) == len(x_test)


#---- Tests for Evaluation metrics ----

@pytest.mark.parametrize("a,b,expected_mse,expected_mae,expected_rmse", [
([1,2,3], [1,2,3], 0, 0, 0),  # perfect prediction
([1,2,3], [2,2,2], 0.67, 0.67, 0.82),  # rough MSE, MAE and RMSE
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

def test_compute_metrics_zero_error():
    """Test the compute_metrics method of Evaluation class."""
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4])
    eval = Evaluation(y_true, y_pred)
    mse, mae, rmse = eval.compute_metrics()
    assert mse == 0
    assert mae == 0
    assert rmse == 0


#---- Tests for WindFarmPlotter ----

def test_plot_predictions_runs_and_saves_image():
    # Create a dummy time series index
    index = pd.date_range(start="2023-01-01", periods=10, freq="h")
    
    # Create dummy y_test and y_pred
    y_test = pd.Series(np.random.rand(10), index=index)
    y_pred = pd.Series(np.random.rand(10), index=index)

    # Create dummy DataFrame for initializing WindFarmPlotter
    dummy_df = pd.DataFrame({"site1": y_test})

    plotter = WindFarmPlotter(dummy_df)

    # Temporary path to save image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        save_path = tmpfile.name

    try:
        plotter.plot_predictions(
            y_test=y_test,
            y_pred=y_pred,
            start_time="2023-01-01 00:00",
            end_time="2023-01-01 09:00",
            title="Prediction Plot Test",
            model_name="TestModel",
            save_path=save_path
        )

        # Check if file was created
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    finally:
        os.remove(save_path)


def test_plot_data_runs_without_error(tmp_path):
    """Test the plot_data method of WindFarmPlotter."""
    plt.switch_backend("Agg")  # Use non-GUI backend

    # Create dummy time-series data
    date_range = pd.date_range(start="2023-01-01", periods=10, freq="h")
    data = pd.DataFrame({"site1": np.random.rand(10)}, index=date_range)

    plotter = WindFarmPlotter(data)

    save_path = str(tmp_path / "test_plot.png")  # Convert to string

    try:
        plotter.plot_data(
            site_index=0,
            column="site1",
            start_time="2023-01-01 00:00",
            end_time="2023-01-01 09:00",
            title="Test Plot",
            save_path=save_path
        )
        assert os.path.exists(save_path)
    except Exception as e:
        pytest.fail(f"Plotting failed with error: {e}")
