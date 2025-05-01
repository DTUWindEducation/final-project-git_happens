"Init file containing functions for final project"
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
#from datetime import datetime
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.metrics import mean_squared_error, mean_absolute_error

#----User input functions----
def get_valid_site_index():
    valid_indices = [1, 2, 3, 4]

    while True:
        try:
            site_index = int(input("Enter the desired location number (1, 2, 3, or 4): "))
            if site_index in valid_indices:
                return site_index
            else:
                print("❌ Invalid input. Please enter a valid value for the location (1, 2, 3, or 4).")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

def get_lag_hours():
    """
    Prompt the user to input the number of lag hours (1-24) and validate the input.

    Returns
    -------
    int
        The number of lag hours specified by the user.
    """
    while True:
        try:
            lag_hours = int(input("Enter the number of lag hours (1-24): "))
            if 1 <= lag_hours <= 24:
                return lag_hours
            else:
                print("❌ Invalid input. Please enter a number between 1 and 24.")
        except ValueError:
            print("❌ Invalid input. Please enter a valid number.")

"""
def get_plot_date():
    
    #Prompt the user to input a date in the format YYYY-MM-DD and validate the input.
    
    
    while True:
        try:
            user_month = int(input("Enter the month you want to plot (01-12): "))
            if not (1 <= user_month <= 12):
                print("❌ Invalid input. Please enter a valid month (01-12).")
                continue

            user_day = int(input(f"Enter the day you want to plot (01-31): "))
            # Validate the date by attempting to create a datetime object
            plot_date = datetime(year=2021, month=user_month, day=user_day)
            return plot_date.strftime('%Y-%m-%d')  # Return the date as a string in 'YYYY-MM-DD' format

        except ValueError:
            print("❌ Invalid date. Please enter a valid day and month.")
"""    
            
# --------------- Week 9 ---------------
class WindFarmDataset:
    def __init__(self, file_path):  #, train_size=0.8, test_size=0.2):
        """
        Initialize the WindFarmDataset with the file path.
        """
        self.file_path = file_path
        self.data = None
        #self.train_size = train_size
        #self.test_size = test_size

    def load_data(self):
        """
        Load the dataset from the specified file path.
        """
        self.data = pd.read_csv(self.file_path, parse_dates=['Time'], index_col='Time', sep=',')
        self.data = self.data.dropna()
        return self.data
    
    def summary(self):
        """
        Print a summary of basic statistics for the dataset.
        """
        return self.data.describe()
    
    def transform_wind_directions(self):
        """
        Transform wind directions from degrees to radians.
        """        
        # Convert wind direction from degrees to radians
        self.data['winddirection_10m'] = np.sin(np.deg2rad(self.data['winddirection_10m']))
        self.data['winddirection_100m'] = np.sin(np.deg2rad(self.data['winddirection_100m']))
        
        return self.data
    '''
    def split(self):
        """
        Split the dataset into training and testing sets.
        """
        train_size = int(len(self.data) * self.train_size)
        test_size = int(len(self.data) * self.test_size)
        
        train_data = self.data.iloc[:train_size]
        test_data = self.data.iloc[train_size:train_size + test_size]
        
        return train_data, test_data
    '''
    """
    def create_lagged_features(self):
            '''
            Creates a new dataset with lagged features based on origin 
            dataset features (1 hour shift, can be changed later).
            '''            
            if self.data is None:
                raise ValueError("Data not loaded. Please load the data first.")
            
            #self.data['Power_lag1'] = self.data['Power'].shift(1)
            #self.data['windspeed_100m_lag1'] = self.data['windspeed_100m'].shift(1)
            
            for col in self.data.columns: # Iterate through each column in the dataset
                self.data[f'{col}_lag_1'] = self.data[col].shift(1) # Create lagged features (1 hour shift)
            
            self.data = self.data.dropna() # Drop missing values from lag/rolling features
            
            return self.data
    """
    
    def split_data(self, lag_hours):
        if self.data is None:
            raise ValueError("Data not loaded. Please load the data first.")
        
        data = self.data.copy()  # Create a copy of the data to avoid modifying the original

        #Create lagged features
        for col in data.columns: # Iterate through each column in the dataset
            data[f'{col}_lag_{lag_hours}'] = data[col].shift(lag_hours) # Create lagged features (1 hour shift)
            
        data = data.dropna() # Drop missing values from lag/rolling features

        # Correlation matrix
        corr_plot_features = ['Power']+[f for f in data.columns if f'_lag_{lag_hours}' in f]
        corr_subset = data[corr_plot_features].corr()
        
        plt.figure(figsize=(8, 4))
        sns.heatmap(corr_subset, cmap="Greens", annot=True)
        plt.title(f'Correlation heatmap (lag = {lag_hours})')
        plt.tight_layout()
        plt.show()
        
        # Feature selection based on correlation threshold
        correlations = corr_subset['Power'].drop('Power')  # Remove self-correlation
        selected_features = correlations[abs(correlations) > 0.5].index.tolist()
        
        print('The features selected for the machine learning model are thus:', selected_features)
        #Split the lagged dataset into training and testing sets (feature and target).
        #Does not shuffle time-series data — future data should not influence past predictions.
        split_idx = int(len(data) * 0.8)  # 80% for training, 20% for testing
    
        #feature_columns = [col for col in data.columns if f'_lag_{lag_hours}' in col]  # Select only columns with lagged features
        X = data[selected_features]  # Selected Features
        y = data['Power']  # 'Power' is the target variable
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test


class WindFarmPlotter:
    def __init__(self, data):
        """
        Initialize the Plotter with the dataset.
        """
        self.data = data

    def plot_data(self, site_index, column, start_time=None, end_time=None, title='Data Plot', xlabel='Time', ylabel='Value', label_legend='Data'):
        """
        Plot the data for a specific site and time range.
        """
        x_data = self.data.loc[start_time:end_time].index
        y_data = self.data.loc[start_time:end_time, column]

        plt.figure(figsize=(8, 4))
        plt.plot(x_data, y_data, label=label_legend)
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.xlim(pd.to_datetime(start_time), pd.to_datetime(end_time))
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(fontsize=10)
        plt.grid()
        plt.show()
        
    def plot_predictions(self, y_test, y_pred, start_time=None, end_time=None, title='Model Predictions vs Actual', model_name='Model', save_path=None):
        
        """
        if user_month in [4, 6, 9, 11]:  # Months with 30 days
            end_date = f"2021-{user_month:02d}-30"
        elif user_month == 2:  # February (assuming non-leap year)
            end_date = f"2021-{user_month:02d}-28"
        else:  # Months with 31 days
            end_date = f"2021-{user_month:02d}-31"
        
        start_date = f"2021-{user_month}-01"
        """
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.loc[start_time:end_time], label='Real', color='red', linewidth=2)
        #plt.plot(y_test[start_date:end_date], label='Real', color='red', linewidth=2)
        
        plt.plot(y_pred.loc[start_time:end_time], label=f'Predicted - {model_name}', linestyle='--', linewidth=2) 
        #plt.plot(y_pred[start_date:end_date], label=f'Predicted - {model_name}', linestyle='--', linewidth=2)
        plt.title(title, fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Power', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300)  # Save the plot as a PNG file
        #plt.show()

class Evaluation:
    def __init__(self,y_true, y_pred):
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
        #if len(self.y_true) != len(self.y_pred):
        #    self.y_true = self.y_true[1:]  # Shift y_true to match the length of y_pred

        mse = mean_squared_error(self.y_true, self.y_pred)
        mae = mean_absolute_error(self.y_true, self.y_pred)
        rmse = np.sqrt(mse)
        
        return mse, mae, rmse

class Prediction:
    def __init__(self, X_test, y_test, X_train, y_train):
        """
        Initialize the Prediction class with the model and data.
        """
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train

    # Persistence Model
    def persistence_model(self, lag_hours):
        """
        Predict one-hour ahead power output using persistence model.
        """
        column_name = f'Power_lag_{lag_hours}'
        if column_name not in self.X_test.columns:
            raise ValueError(f"Column '{column_name}' not found in the dataset. Ensure lagged features are created correctly.")

        y_pred_persistence = self.X_test[column_name]  # Use the lagged column for prediction
        
        #y_pred_persistence = self.X_test['Power_lag_{lag_hours}'] 
        #y_pred_persistence = self.y_test.shift(1)  # one step persistence (1 hour ahead)
        #y_pred_persistence = y_pred_persistence.dropna()  # Drop the first row with NaN value
        
        return y_pred_persistence

    # Linear Regression Model
    def train_linear_regression(self):
        """
        Train a linear regression model.
        """
        model_linear_reg = LinearRegression()
        model_linear_reg.fit(self.X_train, self.y_train)
        
        y_pred_linear_reg = model_linear_reg.predict(self.X_test) # Predict on training data
        y_pred_linear_reg = pd.Series(y_pred_linear_reg, index=self.y_test.index) # Convert to Series with same index as y_test
        
        return y_pred_linear_reg

    # Support Vector Machine (SVM) Model
    def train_svm(self):
        """
        Train a Support Vector Machine (SVM) model.
        """    
        model_svm = svm.SVR()
        model_svm.fit(self.X_train, self.y_train)
        
        y_pred_svm = model_svm.predict(self.X_test)  # Predict on training data
        y_pred_svm = pd.Series(y_pred_svm, index=self.y_test.index)  # Convert to Series with same index as y_test
        
        return y_pred_svm

    # Random Forest Model
    def train_random_forest(self):
        """
        Train a Random Forest model.
        """
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        model_rf.fit(self.X_train, self.y_train)
        
        y_pred_rf = model_rf.predict(self.X_test)  # Predict on test data
        y_pred_rf = pd.Series(y_pred_rf, index=self.y_test.index)  # Convert to Series with same index as y_test
        
        return y_pred_rf