"Main script for final project"
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import src as src 

# Can be modified for the different sites
site_index = 1
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

WindData = src.WindFarmDataset(data_dir, train_size, test_size)
data=WindData.load_data()
print(data.head())      # print data head

# Print statistics of the data
summary = WindData.summary()
print(summary)

# Print lenth of training and testing datasets
split=WindData.split()
print(len(split[0]), len(split[1]))

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
#print('Training Data:', X_train)
#print('Test Data:', X_test)
#print('Training Labels:', y_train)
#print('Test Labels:', y_test)

#Persistence Model
y_pred_persistence = src.persistence_model(y_test)
#print('Persistence Model Predictions:', y_pred_persistence)
#print('length of Persistence Model Predictions:', len(y_pred_persistence))
#print('y_test:', y_test)
#print('length of y_test:', len(y_test))
mse_persistence, mae_persistence, rmse_persistence = src.compute_metrics(y_test[1:], y_pred_persistence)
print(f'Persistence Model - MSE: {mse_persistence}, MAE: {mae_persistence}, RMSE: {rmse_persistence}')

#Linear Regression Model
y_pred_linear_reg = src.train_linear_regression(X_train, X_test, y_train)
mse_lin_reg, mae_lin_reg, rmse_lin_reg = src.compute_metrics(y_test, y_pred_linear_reg)
print(f'Linear Regression Model - MSE: {mse_lin_reg}, MAE: {mae_lin_reg}, RMSE: {rmse_lin_reg}')

#SVM Model
y_pred_svm = src.train_svm(X_train, X_test, y_train)
mse_svm, mae_svm, rmse_svm = src.compute_metrics(y_test, y_pred_svm)
print(f'SVM Model - MSE: {mse_svm}, MAE: {mae_svm}, RMSE: {rmse_svm}')
print('Training Data:', X_train)
print('Test Data:', X_test)
print('Training Labels:', y_train)
print('Test Labels:', y_test)
"""

