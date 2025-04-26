"Main script for final project"
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import src as src 

# Can be modified for the different sites
site_index = 3
train_size=0.8
test_size=0.2

# Load the data for the specified site
data_dir = f'./inputs/Location{site_index}.csv'
WindData = src.WindFarmDataset(data_dir)
data=WindData.load_data()
print(data.head())      # print data head

# Print statistics of the data
summary = WindData.summary()
print(summary)

"""
# Print lenth of training and testing datasets
train_data, test_data = WindData.split()
print ('Training Data:', train_data) 
print ('Test Data:', test_data)
"""

# Plotting the wind speed data
src.plot_data( y_data_input=data['windspeed_100m'],
               site_index=site_index,
               start_time='2020-01-01 00:00:00',
               end_time='2020-01-31 00:00:00',
               title=f'Wind Speed January 2020 at Location {site_index}',
               xlabel='Time',
               ylabel='Wind Speed [m/s]',
               label_legend='Wind Speed at 100m Height'
               )

#Create lagged features
lagged_data= WindData.create_lagged_features()

# Split the lagged dataset into training and testing sets
X_train, X_test, y_train, y_test = src.split_data(lagged_data)
print('Training Data:', X_train)
print('Test Data:', X_test)
print('Training Labels:', y_train)
print('Test Labels:', y_test)