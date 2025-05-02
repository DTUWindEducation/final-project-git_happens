"Main script for final project"
import src as src
import time
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Site location and training/testing data size
# site_index = 1      # Change site index to load different locations
# train_size=0.8      # Proportion of data to be used for training
# test_size=0.2       # Proportion of data to be used for testing


# Example usage
if __name__ == "__main__":
    print("Welcome to the Wind Farm Prediction System!")
    print("This program will help you predict wind farm power generation.")

    site_index = src.get_valid_site_index()
    print(f"✅ You selected location: {site_index}")
    print("⏱️ Timer has been started...")
    start_time = time.time()  # Start the timer


    # Load the data for the specified site
    data_dir = f'./inputs/Location{site_index}.csv'

    # Load wind data
    WindData = src.WindFarmDataset(data_dir)  #1, train_size, test_size)
    data = WindData.load_data()
    WindData.transform_wind_directions()  # Transform wind direction to radians
    #print(data.head())      # print data head

    print(data)
    # Plot windspeedat100m
    wind_speed_plotter = src.WindFarmPlotter(data)
    print("📊 Plotting windspeed at 100m...")
    wind_speed_plotter.plot_data(
        site_index=site_index,
        column= 'windspeed_100m',
        start_time="2017-01-01",
        end_time="2020-12-31",
        title="Windspeed at 100m",
        xlabel="Time",
        ylabel="Windspeed (m/s)",
        label_legend="Windspeed at 100m"
    )
    # Print statistics of the data
    # summary = WindData.summary()
    # print(summary)

    # Get lag hours from the user
    lag_hours = src.get_lag_hours()
    
    # Split data
    X_train, X_test, y_train, y_test = WindData.split_data(lag_hours)

    # Load models
    Model = src.Prediction(X_test, y_test, X_train, y_train)
    y_pred_persistence = Model.persistence_model(lag_hours)
    y_pred_linear_reg = Model.train_linear_regression()
    y_pred_svm = Model.train_svm()
    y_pred_rf = Model.train_random_forest()

    # Evaluate models
    Evaluation = src.Evaluation(y_test, y_pred_persistence)
    mse_persistence, mae_persistence, rmse_persistence = Evaluation.compute_metrics()
    print(f'Persistence Model - MSE: {mse_persistence}, MAE: {mae_persistence}, RMSE: {rmse_persistence}')

    Evaluation = src.Evaluation(y_test, y_pred_linear_reg)
    mse_linear_reg, mae_linear_reg, rmse_linear_reg = Evaluation.compute_metrics()
    print(f'Linear Regression Model - MSE: {mse_linear_reg}, MAE: {mae_linear_reg}, RMSE: {rmse_linear_reg}')

    Evaluation = src.Evaluation(y_test, y_pred_svm)
    mse_svm, mae_svm, rmse_svm = Evaluation.compute_metrics()
    print(f'SVM Model - MSE: {mse_svm}, MAE: {mae_svm}, RMSE: {rmse_svm}')
    
    Evaluation = src.Evaluation(y_test, y_pred_rf)
    mse_rf, mae_rf, rmse_rf = Evaluation.compute_metrics()
    print(f'Random Forest Model - MSE: {mse_rf}, MAE: {mae_rf}, RMSE: {rmse_rf}')
    
    # Get the date to plot
    #plot_date = src.get_plot_date()
        
    Plotter = src.WindFarmPlotter(data)
    #Plotter.plot_predictions(y_test, y_pred_persistence, y_pred_linear_reg, y_pred_svm)

    Plotter.plot_predictions(y_test, y_pred_persistence, start_time="2021-12-24", end_time="2021-12-31", model_name='Persistence Model', save_path=os.path.join('./outputs/', f'Persistence_Model_Location{site_index}.png'))
    Plotter.plot_predictions(y_test, y_pred_linear_reg, start_time="2021-12-24", end_time="2021-12-31", model_name='Linear Regression', save_path=os.path.join('./outputs/', f'Linear_Regression_Location{site_index}.png'))
    Plotter.plot_predictions(y_test, y_pred_svm, start_time="2021-12-24", end_time="2021-12-31", model_name='SVM', save_path=os.path.join('./outputs/', f'SVM_Location{site_index}.png'))
    Plotter.plot_predictions(y_test, y_pred_rf, start_time="2021-12-24", end_time="2021-12-31", model_name='Random Forest', save_path=os.path.join('./outputs/', f'Random_Forest_Location{site_index}.png'))

    print("📊 Generated plots are saved in the ./output/ directory")
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    print(f"✅ Script finished in {elapsed_time:.2f} seconds.")
