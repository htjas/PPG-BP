import wfdb
from matplotlib import pyplot as plt
import numpy as np
from scripts import *
import pandas as pd
from jproperties import Properties
from visual import *

configs = Properties()

# Load project details
with open('details.properties', 'rb') as config_file:
    configs.load(config_file)

# Init (Fetching and Filtering)

# database_name = configs.get('mimic4').data
# load_filter_save_records(database_name)

# After Init
print("Reading matching_records.csv")
df = pd.read_csv('matching_records.csv')
# All segments array
segments = df.values

# get 10 minutes of bp and ppg data, and save to /data folder
# extract_save_bp_ppg_data(segments, configs.get('path_of_data').data)