"Main script for final project"
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import src as src 

data_dir = './inputs/Location1.csv'
data=src.load_data(data_dir)
print(data)