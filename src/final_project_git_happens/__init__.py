"Init file containing functions for final project"
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ----User input functions----


def get_valid_site_index():
    """
    Prompt the user to input a valid site index and return the selected index.
    This function prompts the user to enter a location number
    until a valid input is provided. Valid inputs are integers from the
    predefined list of valid indices [1, 2, 3, 4]. If the user enters an
    invalid value or a non-integer, an appropriate error message is displayed.
    Returns:
        int: The valid site index selected by the user.
    """
    valid_indices = [1, 2, 3, 4]

    while True:
        try:
            site_index = int(
                input("Enter the desired location number (1, 2, 3, or 4): ")
            )
            if site_index in valid_indices:
                return site_index
            else:
                print(
                    "❌ Invalid input. Please enter a valid value for the "
                    "location (1, 2, 3, or 4)."
                )
        except ValueError:
            print("❌ Invalid input. Please enter a number.")


def get_lag_hours():
    """
    Prompt the user to input the number of lag hours (1 or 24) and validate the
    input.

    Returns
    -------
    int
        The number of lag hours specified by the user.
    """
    while True:
        try:
            lag_hours = int(input("Enter the desired forecasting horizon (1 or 24): "))
            if lag_hours == 1 or lag_hours == 24:
                print(f"✅ You selected a forecasting horizon of {lag_hours} hour(s).")
                return lag_hours
            else:
                print(
                    "❌ Invalid input. Please enter a number between 1 and 24."
                )
        except ValueError:
            print("❌ Invalid input. Please enter a valid number.")


class WindFarmDataset:
    """
    This class handles and preprocesss wind farm data for machine learning
    and data analysis. It provides the ability to load data, transform wind
    direction features, create lagged features, and split the data into
    training and testing sets. The class also includes methods for visualizing
    correlations between features and the target variable.

    Attributes:
        file_path (str): The file path to the dataset.
        data (pd.DataFrame): The loaded dataset.

    Methods:
        load_data():
            Loads the dataset from the specified file path, parses dates, and
            removes missing values.
        summary():
            Returns a summary of basic statistics for the dataset.
        transform_wind_directions():
            Transforms wind direction features from degrees to radians.
        split_data(lag_hours):
            Splits the dataset into training and testing sets after creating
            lagged features and selecting relevant features based on
            correlation with the target variable.
    """

    def __init__(self, file_path):  # train_size=0.8, test_size=0.2):
        """
        Initialize the WindFarmDataset with the file path.
        """
        self.file_path = file_path
        self.data = None


    def load_data(self):
        """
        Load the dataset from the specified file path.
        """
        self.data = pd.read_csv(
            self.file_path, parse_dates=['Time'], index_col='Time', sep=','
        )
        self.data = self.data.dropna()
        return self.data

    def transform_wind_directions(self):
        """
        Transform wind directions from degrees to radians.
        """
        # Convert wind direction from degrees to radians
        self.data['winddirection_10m'] = np.sin(
            np.deg2rad(self.data['winddirection_10m'])
        )
        self.data['winddirection_100m'] = np.sin(
            np.deg2rad(self.data['winddirection_100m'])
        )
        return self.data


    def split_data(self, lag_hours, split_index):
        """
        Splits the dataset into training and testing sets after creating
        lagged features and selecting relevant features based on correlation
        with the target variable.

        Args:
            lag_hours (int): The number of hours to lag the features for
            creating lagged variables.
            split_index (float): The proportion of the dataset to include in
            the training set (e.g., 0.8 for 80% training and 20% testing).

        Raises:
            ValueError: If the data attribute is not loaded before calling
            this method.

        Returns:
            tuple: A tuple containing:
                - x_train (pd.DataFrame): Training set features.
                - x_test (pd.DataFrame): Testing set features.
                - y_train (pd.Series): Training set target variable.
                - y_test (pd.Series): Testing set target variable.
        """

        # Create a copy of the data to avoid modifying the original
        data = self.data.copy()

        # Create lagged features
        for col in data.columns:  # Iterate through each column in the dataset
            data[f'{col}_lag_{lag_hours}'] = data[col].shift(lag_hours)
            # Create lagged features (1 hour shift)

        data = data.dropna()  # Drop missing values from lag/rolling features

        # Correlation matrix for feature selection
        corr_plot_features = ['Power'] + [
            f for f in data.columns if f'_lag_{lag_hours}' in f
        ]
        corr_subset = data[corr_plot_features].corr()

        # Plotting the correlation heatmap
        plt.figure(figsize=(8, 4))
        sns.heatmap(corr_subset, cmap="Greens", annot=True)
        plt.title(f'Correlation heatmap (lag = {lag_hours})')
        plt.tight_layout()
        plt.savefig(
            f'./outputs/correlation_heatmap_lag_{lag_hours}.png',
            dpi=300
        )

        # Feature selection based on correlation threshold
        correlations = corr_subset['Power'].drop('Power')  # Remove self-correlation
        # for 1 hour ahead prediction, select features with correlation > 0.5
        if lag_hours == 1:
            selected_features = correlations[abs(correlations) > 0.5].index.tolist()
        # for 24 hours ahead prediction, select features with correlation > 0.15
        elif lag_hours == 24:
            selected_features = correlations[abs(correlations) > 0.15].index.tolist()
        # printing the selected features
        print(
            'ℹ️ The features selected for the machine learning model are thus:',
            selected_features
        )
        # Split the lagged dataset into training and testing sets
        # (feature and target).
        # Does not shuffle time-series data — future data should not influence
        # past predictions.
        split_idx = int(len(data) * split_index)  # split_index defined in main.py

        x = data[selected_features]  # Selected Features
        y = data['Power']  # 'Power' is the target variable

        # creating train and test set based on feature and split ratio
        x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        return x_train, x_test, y_train, y_test


class WindFarmPlotter:
    """
    This class provides methods to plot time-series data for specific sites
    and time ranges, as well as to compare actual and predicted values from
    machine learning models.

    Attributes:
        data (pd.DataFrame): The dataset containing wind farm data, indexed by
        time.

    Methods:
        plot_data(
            site_index, column, start_time=None, end_time=None,
            title='Data Plot', xlabel='Time', ylabel='Value',
            label_legend='Data', save_path
        ):
            Plots the time-series data for a specific site and time range.

        plot_predictions(
            y_test, y_pred, start_time=None, end_time=None,
            title='Model Predictions vs Actual', model_name='Model',
            save_path=None
        ):
            Plots the comparison between actual and predicted values for a
            given time range.
    """

    def __init__(self, data):
        """
        Initialize the Plotter with the dataset.
        """
        self.data = data

    def plot_data(
        self, site_index, column, start_time=None, end_time=None,
        title='Data Plot', xlabel='Time', ylabel='Value', label_legend='Data', save_path=None
    ):
        """
        Plot the data for a specific site and time range.
        Args:
            site_index (int): The index of the site to plot.
            column (str): The column name to plot.
            start_time (str): The start time for the plot (optional).
            end_time (str): The end time for the plot (optional).
            title (str): The title of the plot.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            label_legend (str): The legend label for the plot.
            save_path (str): The path to save the plot image.
        """
        x_data = self.data.loc[start_time:end_time].index
        y_data = self.data.loc[start_time:end_time, column]

        plt.figure(figsize=(8, 5))
        plt.subplots_adjust(bottom=0.2)  # Add more vertical room for x-axis labels
        plt.plot(x_data, y_data, label=label_legend)
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.xlim(pd.to_datetime(start_time), pd.to_datetime(end_time))
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(fontsize=10)
        plt.grid()
        plt.savefig(save_path, dpi=300)  # Save the plot as a PNG file


    def plot_predictions(
        self, y_test, y_pred, start_time=None, end_time=None,
        title='Model Predictions vs Actual', model_name='Model', save_path=None
    ):
        """
        Plot the comparison between actual and predicted values for a given
        location and time range.
        Args:
            y_test (pd.Series): The actual values (ground truth).
            y_pred (pd.Series): The predicted values from the model.
            start_time (str): The start time for the plot.
            end_time (str): The end time for the plot.
            title (str): The title of the plot.
            model_name (str): The name of the model used for predictions.
            save_path (str): The path to save the plot image.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(
            y_test.loc[start_time:end_time],
            label='Real',
            color='red',
            linewidth=2
        )
        # plt.plot(y_test[start_date:end_date], label='Real', color='red', linewidth=2)

        plt.plot(
            y_pred.loc[start_time:end_time],
            label=f'Predicted - {model_name}',
            linestyle='--',
            linewidth=2)
        # plt.plot(y_pred[start_date:end_date], label=f'Predicted - {model_name}', linestyle='--', linewidth=2)
        plt.title(title, fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Power', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)  # Save the plot as a PNG file

class Evaluation:
    """
    This class provides values to evaluate the performance of predictive models
    by calculating MSE, MAE, and RMSE.

    Attributes
    y_true : array-like
        The ground truth (true) values.
    y_pred : array-like
        The predicted values.

    Methods
    compute_metrics():
        Computes and returns the MSE, MAE, and RMSE metrics based on the true
        and predicted values.
    """

    def __init__(self, y_true, y_pred):
        """
        Initialize the Evaluation class with true and predicted values.
        """
        self.y_true = y_true
        self.y_pred = y_pred

    # ---Compute Metrics---
    def compute_metrics(self):
        """
        Compute and return the evaluation metrics.

        Parameters
        ----------
        y_true : numpy.ndarray
            True values.
        y_pred : numpy.ndarray
            Predicted values.

        Returns
        -------
        RMSE, MAE, and MSE.
        """
        # Calculate evaluation metrics
        mse = round(mean_squared_error(self.y_true, self.y_pred),5)
        mae = round(mean_absolute_error(self.y_true, self.y_pred),5)
        rmse = round(np.sqrt(mse),5)

        return mse, mae, rmse


class Prediction:
    """
    This class predicts power output using various machine learning models,
    including a persistence model, linear regression, support vector machine
    (SVM), and random forest. It is designed to work with time-series data,
    where lagged features are used for prediction.

    Attributes:
        x_test (pd.DataFrame): The test dataset features.
        y_test (pd.Series): The test dataset target variable.
        x_train (pd.DataFrame): The training dataset features.
        y_train (pd.Series): The training dataset target variable.

    Methods:
        persistence_model(lag_hours):
            Predicts power output using a persistence model based on lagged
            features.
        train_linear_regression():
            Trains a linear regression model and predicts power output on the
            test dataset.
        train_svm():
            Trains a Support Vector Machine (SVM) model and predicts power
        output on the test dataset.
        train_random_forest():
            Trains a Random Forest model and predicts power output on the test
            dataset.
    """

    def __init__(self, x_test, y_test, x_train, y_train):
        """
        Initialize the Prediction class with the model and data.
        """
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train

    # Persistence Model
    def persistence_model(self, lag_hours):
        """
        Predict one-hour ahead power output using persistence model.
        The persistence model uses the lagged feature of the target variable
        to predict the next value.
        """
        column_name = f'Power_lag_{lag_hours}'
        # Use the lagged column for prediction
        y_pred_persistence = self.x_test[column_name]

        # y_pred_persistence = self.x_test['Power_lag_{lag_hours}']
        # y_pred_persistence = self.y_test.shift(1)  # one step persistence (1 hour ahead)
        # y_pred_persistence = y_pred_persistence.dropna()  # Drop the first row with NaN value

        return y_pred_persistence

    # Linear Regression Model
    def train_linear_regression(self):
        """
        Train a linear regression model.
        """
        model_linear_reg = LinearRegression()
        model_linear_reg.fit(self.x_train, self.y_train)

        # Predict on training data
        y_pred_linear_reg = model_linear_reg.predict(self.x_test)

        # Convert to Series with same index as y_test
        y_pred_linear_reg = pd.Series(
            y_pred_linear_reg, index=self.y_test.index
        )

        return y_pred_linear_reg

    # Support Vector Machine (SVM) Model
    def train_svm(self):
        """
        Train a Support Vector Machine (SVM) model.
        """
        model_svm = svm.SVR()
        model_svm.fit(self.x_train, self.y_train)

        # Predict on training data
        y_pred_svm = model_svm.predict(self.x_test)

        # Convert to Series with same index as y_test
        y_pred_svm = pd.Series(y_pred_svm, index=self.y_test.index)

        return y_pred_svm

    # Random Forest Model
    def train_random_forest(self):
        """
        Train a Random Forest model.
        """
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        model_rf.fit(self.x_train, self.y_train)

        # Predict on test data
        y_pred_rf = model_rf.predict(self.x_test)

        # Convert to Series with same index as y_test
        y_pred_rf = pd.Series(y_pred_rf, index=self.y_test.index)

        return y_pred_rf
