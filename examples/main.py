"Main script for final project"
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import src as src 

# Site location and training/testing data size
site_index = 1      # Change site index to load different locations
train_size=0.8      # Proportion of data to be used for training
test_size=0.2       # Proportion of data to be used for testing

# Load the data for the specified site
data_dir = f'./inputs/Location{site_index}.csv'

# Load wind data
WindData = src.WindFarmDataset(data_dir, train_size, test_size)
data = WindData.load_data()
print(data.head())      # print data head

# Print statistics of the data
# summary = WindData.summary()
# print(summary)

# Split data
X_train, X_test, y_train, y_test = WindData.split_data()

# Load models
Model = src.Prediction(X_test, y_test, X_train, y_train)
y_pred_persistence = Model.persistence_model()
y_pred_linear_reg = Model.train_linear_regression()
y_pred_svm = Model.train_svm()

# Evaluate models
Evaluation = src.Evaluation(y_test, y_pred_persistence)
mse_persistence, mae_persistence, rmse_persistence = Evaluation.compute_metrics()
print(f'Persistence Model - MSE: {mse_persistence}, MAE: {mae_persistence}, RMSE: {rmse_persistence}')

Evaluation = src.Evaluation(y_test, y_pred_linear_reg)
mse_linear_reg, mae_linear_reg, rmse_linear_reg = Evaluation.compute_metrics()
print(f'Linear Regression Model - MSE: {mse_linear_reg}, MAE: {mae_linear_reg}, RMSE: {rmse_linear_reg}')

Evaluation = src.Evaluation(y_test, y_pred_svm)
rms_svm, mae_svm, mse_svm = Evaluation.compute_metrics()
print(f'SVM Model - MSE: {mse_svm}, MAE: {mae_svm}, RMSE: {rms_svm}')