"Main script for final project"
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import src as src 

# Can be modified for the different sites
site_index = 3

# Load the data for the specified site
data_dir = f'./inputs/Location{site_index}.csv'
data=src.load_data(data_dir)

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