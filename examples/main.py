"Main script for final project"
import time
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import src as src

#Initialising split index
split_index = 0.8 # 80% train, 20% test

# This is the main script for the wind farm prediction system.

if __name__ == "__main__":
    print("Welcome to the Wind Farm Prediction System!")
    print("This program will help you predict wind farm power generation.")

    # User chooses a location
    site_index = src.get_valid_site_index()
    print(f"✅ You selected location: {site_index}")
    print("⏱️ Timer has been started...")
    start_time = time.time()  # Start the timer

    # Load the data for the specified site
    DATA_DIR = f'./inputs/Location{site_index}.csv'

    # Load wind data
    WindData = src.WindFarmDataset(DATA_DIR)  # 1, train_size, test_size)
    data = WindData.load_data()
    WindData.transform_wind_directions()  # Transform wind direction to radians

    # Visualising the wind speed data
    Plotter = src.WindFarmPlotter(data)
    print("📊 Plot of wind speeds at 100m can be seen in the ./outputs/ directory.")
    Plotter.plot_data(
        site_index=site_index,
        column='windspeed_100m',
        start_time="2017-01-01",
        end_time="2020-12-31",
        title=f"Location {site_index}: Windspeed at 100m",
        xlabel="Time",
        ylabel="Windspeed (m/s)",
        label_legend="Windspeed at 100m",
        save_path=os.path.join(
            "./outputs/", f"Windspeed_100m_Location{site_index}.png")
    )

    # User chooses the desired forecasting horizon
    lag_hours = src.get_lag_hours()

    # Split data into train and test data based on the split index and lag hours
    x_train, x_test, y_train, y_test = WindData.split_data(lag_hours, split_index)

    # Letting user know that forecasts are being generated
    print("⏳ Generating forecasts...")

    # Each model is loaded and trained
    Model = src.Prediction(x_test, y_test, x_train, y_train)
    y_pred_persistence = Model.persistence_model(lag_hours)
    y_pred_linear_reg = Model.train_linear_regression()
    y_pred_svm = Model.train_svm()
    y_pred_rf = Model.train_random_forest()

    # Evaluate models and print metrics

    #persistence model
    evaluation_persistence = src.Evaluation(y_test, y_pred_persistence)
    mse_persistence, mae_persistence, rmse_persistence = (
        evaluation_persistence.compute_metrics()
    )
    print("📝 Results of the models: ")
    print(
        f'Persistence Model - MSE: {mse_persistence}, '
        f'MAE: {mae_persistence}, RMSE: {rmse_persistence}'
    )

    # Linear regression model
    evaluation_linear = src.Evaluation(y_test, y_pred_linear_reg)
    mse_linear_reg, mae_linear_reg, rmse_linear_reg = (
        evaluation_linear.compute_metrics()
    )
    print(
        f'Linear Regression Model - MSE: {mse_linear_reg}, '
        f'MAE: {mae_linear_reg}, RMSE: {rmse_linear_reg}'
    )

    # SVM model
    evaluation_svm = src.Evaluation(y_test, y_pred_svm)
    mse_svm, mae_svm, rmse_svm = evaluation_svm.compute_metrics()
    print(f'SVM Model - MSE: {mse_svm}, MAE: {mae_svm}, RMSE: {rmse_svm}')

    # Random Forest model
    evaluation_rf = src.Evaluation(y_test, y_pred_rf)
    mse_rf, mae_rf, rmse_rf = evaluation_rf.compute_metrics()
    print(
        f'Random Forest Model - MSE: {mse_rf}, '
        f'MAE: {mae_rf}, RMSE: {rmse_rf}'
    )

    # Plot predictions for each model
    Plotter.plot_predictions(
        y_test,
        y_pred_persistence,
        start_time="2021-12-24",
        end_time="2021-12-31",
        model_name="Persistence Model",
        save_path=os.path.join(
            "./outputs/", f"Persistence_Model_Location{site_index}.png"
        ),
    )
    Plotter.plot_predictions(
        y_test,
        y_pred_linear_reg,
        start_time="2021-12-24",
        end_time="2021-12-31",
        model_name="Linear Regression",
        save_path=os.path.join(
            "./outputs/", f"Linear_Regression_Location{site_index}.png"
        ),
    )
    Plotter.plot_predictions(
        y_test,
        y_pred_svm,
        start_time="2021-12-24",
        end_time="2021-12-31",
        model_name="SVM",
        save_path=os.path.join(
            "./outputs/", f"SVM_Location{site_index}.png"
        ),
    )
    Plotter.plot_predictions(
        y_test,
        y_pred_rf,
        start_time="2021-12-24",
        end_time="2021-12-31",
        model_name="Random Forest",
        save_path=os.path.join(
            "./outputs/", f"Random_Forest_Location{site_index}.png"
        ),
    )

    # Letting user know where to find the plots
    print("📊 Generated plots are saved in the ./output/ directory")
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    print(f"✅ Script finished in {elapsed_time:.2f} seconds.")
