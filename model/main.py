from init_scripts import load_filter_and_save_records
from sp_scripts import process_data
from ml_scripts import run_model
from jproperties import Properties
import pandas as pd
import os


# Load project details
configs = Properties()
with open('details.properties', 'rb') as config_file:
    configs.load(config_file)

#
# Load records and extract maximum median values from MIMIC4 features (Validation dataset: x - 20%)
#

# Loading, Filtering and Saving Records
# db_mm4 = configs.get('mimic4_url').data
# path_mm4 = configs.get('path_of_mimic4_data').data
# load_filter_and_save_records(db_mm4, path_mm4, records_to_load=999999, single_record_arrays=999999)

# # Signal processing
# fs_mm4 = float(configs.get('fs_mimic4').data)
# goal_mm4 = configs.get('goal_mimic4').data
# process_data(fs_mm4, path_mm4, goal_mm4, 7)

#
# Load records and extract median values from MIMIC3 features (Training and Testing dataset: 4x - 80%)
#

fs_mm3 = float(configs.get('fs_mimic3').data)
goal_mm3 = configs.get('goal_mimic3').data
goal_mm4 = configs.get('goal_mimic4').data
db_mm3 = configs.get('mimic3_url').data
path_mm3 = configs.get('path_of_mimic3_data').data

# abs_path = os.path.abspath(os.getcwd())
# mm4_values_size = len(pd.read_csv(f'{abs_path}/features/{goal_mm4}/tot_med_ppg_feats.csv'))
# mm3_values_size = len(pd.read_csv(f'{abs_path}/features/{goal_mm3}/tot_med_ppg_feats.csv'))

# while mm4_values_size * 4 > mm3_values_size:
# Loading, Filtering and Saving Records
load_filter_and_save_records(db_mm3, path_mm3, records_to_load=2500, single_record_arrays=100)
# Signal processing
process_data(fs_mm3, path_mm3, goal_mm3, 7)

#
# # Machine Learning
# run_model(target='sys')  # training-testing
#
# # validate_model() - validation
