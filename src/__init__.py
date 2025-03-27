"Init file containing functions for final project"
import matplotlib.pyplot as plt
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
    





