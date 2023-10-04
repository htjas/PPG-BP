import wfdb
from matplotlib import pyplot as plt
import numpy as np
from setup import *
from jproperties import Properties

configs = Properties()

with open('details.properties', 'rb') as config_file:
    configs.load(config_file)

database_name = configs.get('mimic4').data

records = load_records(database_name)

for record in records:
    print(record.name)
    record_dir = f'{database_name}/{record.parent}'
    record_data = wfdb.rdheader(record.name, pn_dir=record_dir, rd_segments=True)
    print(record_data.sig_name)

