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
median_interval = 7
# process_data(fs_mm4, path_mm4, goal_mm4, median_interval)

#
# Load records and extract median values from MIMIC3 features (Training and Testing dataset: 4x - 80%)
#

fs_mm3 = float(configs.get('fs_mimic3').data)
goal_mm3 = configs.get('goal_mimic3').data
goal_mm4 = configs.get('goal_mimic4').data
db_mm3 = configs.get('mimic3_url').data
path_mm3 = configs.get('path_of_mimic3_data').data

# abs_path = os.path.abspath(os.getcwd())
# n_fetched_records = len(os.listdir(abs_path + path_mm3 + 'abp'))

# mm4_values_size = len(pd.read_csv(f'{abs_path}/features/{goal_mm4}/med_ppg_feats7.csv'))
# mm3_values_size = len(pd.read_csv(f'{abs_path}/features/{goal_mm3}/med_ppg_feats7.csv'))

# while mm4_values_size * 4 > mm3_values_size:
# Loading, Filtering and Saving Records
load_filter_and_save_records(db_mm3, path_mm3, records_to_load=22032, single_record_arrays=100)
# Signal processing
# process_data(fs_mm3, path_mm3, goal_mm3, median_interval)

#
# # Machine Learning\
#
path_med_abp_feats = configs.get(f'abp_median{median_interval}_features_path').data
path_med_ppg_feats = configs.get(f'ppg_median{median_interval}_features_path').data

run_model(path_med_abp_feats, path_med_ppg_feats)  # training-testing
#
# validate_model() - validation
