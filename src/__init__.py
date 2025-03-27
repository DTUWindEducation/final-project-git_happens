"Init file containing functions for final project"

import pandas as pd
import numpy as np


def load_data(data_path):
    """
    Load data from a file and return it as a numpy array.
    
    Parameters
    ----------
    data_path : str
        Path to the data file.
    
    Returns
    -------
    numpy.ndarray
        Data from the file.
    """
    data = pd.read_csv(data_path, parse_dates=['Time'], index_col='Time',sep=',') #reading the data
    data = data.dropna() #dropping the missing values

    return data







