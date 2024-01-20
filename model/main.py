from init_scripts import load_filter_and_save_records
from sp_scripts import process_data
from ml_scripts import run_model
from jproperties import Properties

configs = Properties()

# Load project details
with open('details.properties', 'rb') as config_file:
    configs.load(config_file)

# Loading, Filtering and Saving Records
db_mm3 = configs.get('mimic3_url').data
path_mm3 = configs.get('path_of_mimic3_data').data
load_filter_and_save_records(db_mm3, path_mm3, 300)

# Signal processing
fs_mm3 = float(configs.get('fs_mimic3').data)
goal_mm3 = configs.get('goal_mimic3').data
process_data(fs_mm3, path_mm3, goal_mm3)

# Machine Learning
run_model(target='sys')
