import wfdb
from matplotlib import pyplot as plt
import numpy as np
from setup import *
from jproperties import Properties

configs = Properties()

""" Load project details """
with open('details.properties', 'rb') as config_file:
    configs.load(config_file)
database_name = configs.get('mimic4').data

""" Load all records from database """
records = load_records(database_name)

""" 
Filter records according to criteria:
    - ABP and Pleth signals present
    - 10 min of continuous signal available  
"""
matching_records = filter_records(records, database_name)

""" Save filtered records to .csv File """
save_records_to_csv(matching_records)
