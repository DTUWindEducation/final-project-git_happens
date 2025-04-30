"Main script for final project"
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import src as src 
import time

# Site location and training/testing data size
#site_index = 1      # Change site index to load different locations
#train_size=0.8      # Proportion of data to be used for training
#test_size=0.2       # Proportion of data to be used for testing

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

# Example usage
if __name__ == "__main__":
    print("Welcome to the Wind Farm Prediction System!")
    print("This program will help you predict wind farm power generation.")


    site_index = get_valid_site_index()
    print(f"✅ You selected location: {site_index}")
    print("⏱️ Timer has been started...")
    start_time = time.time()  # Start the timer


    # Load the data for the specified site
    data_dir = f'./inputs/Location{site_index}.csv'

    # Load wind data
    WindData = src.WindFarmDataset(data_dir)  #1, train_size, test_size)
    data = WindData.load_data()
    #print(data.head())      # print data head

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

    # Split data
    X_train, X_test, y_train, y_test = WindData.split_data()

    # Load models
    Model = src.Prediction(X_test, y_test, X_train, y_train)
    y_pred_persistence = Model.persistence_model()
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
    
    while True:
        try:
            user_month = int(input("Enter the month you want to plot (01-12): "))
            if 1 <= user_month <= 12:
                break
            else:
                print("❌ Invalid input. Please enter a number between 1 and 12.")
        except ValueError:
            print("❌ Invalid input. Please enter a valid number.")
        
    Plotter = src.WindFarmPlotter(data)  # or pass `data` if needed
    #Plotter.plot_predictions(y_test, y_pred_persistence, y_pred_linear_reg, y_pred_svm)

    Plotter.plot_predictions(y_test, y_pred_persistence, user_month, model_name='Persistence Model')
    Plotter.plot_predictions(y_test, y_pred_linear_reg, user_month, model_name='Linear Regression')
    Plotter.plot_predictions(y_test, y_pred_svm, user_month, model_name='SVM')
    Plotter.plot_predictions(y_test, y_pred_rf, user_month, model_name='Random Forest')

    print("📊 Generated plots are saved under ......")
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    print(f"✅ Script finished in {elapsed_time:.2f} seconds.")
