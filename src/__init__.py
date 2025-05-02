"Init file containing functions for final project"
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --------------- Week 9 ---------------

class WindFarmDataset:
    def __init__(self, file_path, train_size=0.8, test_size=0.2):
        """
        Initialize the WindFarmDataset with the file path.
        """
        self.file_path = file_path
        self.data = None
        self.train_size = train_size
        self.test_size = test_size

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

    def create_lagged_features(self):
            """
            Creates a new dataset with lagged features based on origin 
            dataset features (1 hour shift, can be changed later).
            """            
            #if self.data is None:
            #    raise ValueError("Data not loaded. Please load the data first.")
            
            #self.data['Power_lag1'] = self.data['Power'].shift(1)
            #self.data['windspeed_100m_lag1'] = self.data['windspeed_100m'].shift(1)
            
            for col in self.data.columns: # Iterate through each column in the dataset
                self.data[f'{col}_lag_1'] = self.data[col].shift(1) # Create lagged features (1 hour shift)
            
            self.data = self.data.dropna() # Drop missing values from lag/rolling features
            
            return self.data

    def split_data(self):
        #if self.data is None:
        #        raise ValueError("Data not loaded. Please load the data first.")
        
        data = self.data.copy()  # Create a copy of the data to avoid modifying the original

        for col in data.columns: # Iterate through each column in the dataset
                data[f'{col}_lag_1'] = data[col].shift(1) # Create lagged features (1 hour shift)
            
        data = data.dropna() # Drop missing values from lag/rolling features

        #Split the lagged dataset into training and testing sets (feature and target).
        #Does not shuffle time-series data — future data should not influence past predictions.
        split_idx = int(len(data) * 0.8)  # 80% for training, 20% for testing
    
        feature_columns = [col for col in data.columns if '_lag_1' in col]  # Select only columns with lagged features
        X = data[feature_columns]  # Features (lagged variables)
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

    def plot_data(self, site_index, start_time, end_time, title='Data Plot', xlabel='Time', ylabel='Value', label_legend='Data'):
        """
        Plot the data for a specific site and time range.
        """
        x_data = self.data.loc[start_time:end_time].index
        y_data = self.data.loc[start_time:end_time].values

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
        if len(self.y_true) != len(self.y_pred):
            self.y_true = self.y_true[1:]  # Shift y_true to match the length of y_pred

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
    def persistence_model(self):
        """
        Predict one-hour ahead power output using persistence model.
        """
        y_pred_persistence = self.y_test.shift(1)  # one step persistence (1 hour ahead)
        y_pred_persistence = y_pred_persistence.dropna()  # Drop the first row with NaN value
        
        return y_pred_persistence

    # Linear Regression Model
    def train_linear_regression(self):
        """
        Train a linear regression model.
        """
        model_linear_reg = LinearRegression()
        model_linear_reg.fit(self.X_train, self.y_train)
        
        y_pred_linear_reg = model_linear_reg.predict(self.X_test) # Predict on training data
        
        return y_pred_linear_reg

    # Support Vector Machine (SVM) Model
    def train_svm(self):
        """
        Train a Support Vector Machine (SVM) model.
        """    
        model_svm = svm.SVR()
        model_svm.fit(self.X_train, self.y_train)
        
        y_pred_svm = model_svm.predict(self.X_test)  # Predict on training data
        
        return y_pred_svm

