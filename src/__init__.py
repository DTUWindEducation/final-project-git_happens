"Init file containing functions for final project"
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --------------- Week 9 ---------------
class WindFarmDataset:
    def __init__(self, file_path):

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

    def split(self):
        """
        Split the dataset into training and testing sets.
        """
        train_size = int(len(self.data) * self.train_size)
        test_size = int(len(self.data) * self.test_size)
        
        train_data = self.data.iloc[:train_size]
        test_data = self.data.iloc[train_size:train_size + test_size]
        
        return train_data, test_data

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

# --------------- Week 8 ---------------

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
    
    def create_lagged_features(self):
        """
        Creates a new dataset with lagged features based on origin 
        dataset features (1 hour shift, can be changed later).
        """
        
        if self.data is None:
            raise ValueError("Data not loaded. Please load the data first.")
        
        #self.data['Power_lag1'] = self.data['Power'].shift(1)
        #self.data['windspeed_100m_lag1'] = self.data['windspeed_100m'].shift(1)
        
        for col in self.data.columns: # Iterate through each column in the dataset
            self.data[f'{col}_lag_1'] = self.data[col].shift(1) # Create lagged features (1 hour shift)
        
        self.data = self.data.dropna() # Drop missing values from lag/rolling features
        
        return self.data
    
    """
    def split(self):
        
        Split the dataset into training and testing sets.
        
        train_size = int(len(self.data) * self.train_size)
        test_size = int(len(self.data) * self.test_size)
        
        train_data = self.data.iloc[:train_size]
        test_data = self.data.iloc[train_size:train_size + test_size]
        
        return train_data, test_data
    """
    
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

# --------------- Week 8 ---------------
"""
def load_data(data_path):
    
    Load data from a file and return it as a numpy array.
    
    Parameters
    ----------
    data_path : str
        Path to the data file.
    
    Returns
    -------
    numpy.ndarray
        Data from the file.

    data = pd.read_csv(data_path, parse_dates=['Time'], index_col='Time',sep=',') #reading the data
    data = data.dropna() #dropping the missing values

    return data
"""

def plot_data(y_data_input,
              site_index,
              start_time,
              end_time,
              title='Data Plot',
              xlabel='Time',
              ylabel='Value',
              label_legend='Data'
              ):
    """
    Plot the data using matplotlib.
    
    Parameters
    ----------
    x_data_input : pandas.Series
        The data to plot on x-axis.
    y_data_input : pandas.Series
        The data to plot on y-axis.
    site_index : int
        The index of the site to plot.
    start_time : str
        Start time for the plot in 'YYYY-MM-DD HH:MM:SS' format.
    end_time : str
        End time for the plot in 'YYYY-MM-DD HH:MM:SS' format.
    title : str, optional
        Title of the plot (default is 'Data Plot').
    xlabel : str, optional
        Label for the x-axis (default is 'Time').
    ylabel : str, optional
            Label for the y-axis (default is 'Value').
    
            
    Returns
    -------
    None
    """
   # data_to_plot = load_data(f'./inputs/Location{site_index}.csv')
    x_data = y_data_input.loc[start_time:end_time].index
    y_data = y_data_input.loc[start_time:end_time].values

    plt.figure(figsize=(8, 4))
    plt.plot(x_data, y_data,label=label_legend)
    plt.title(title,fontsize=16)
    plt.xlabel(xlabel,fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.xlim(pd.to_datetime(start_time), pd.to_datetime(end_time))
    plt.ylabel(ylabel,fontsize=12)
    #plt.ylim(np.min(y_data), np.max(y_data))
    plt.legend(fontsize=10)
    plt.grid()
    plt.show()


# --- Split Data --- 
def split_data(data):
    """Split the lagged dataset into training and testing sets (feature and target)."""
    feature_columns = [col for col in data.columns if '_lag_1' in col]  # Select only columns with lagged features
    X = data[feature_columns]  # Features (lagged variables)
    y = data['Power']  # 'Power' is the target variable
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    return X_train, X_test, y_train, y_test


# ---Compute Metrics---
def compute_metrics(y_true, y_pred):
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
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return mse, mae, rmse
