import wfdb
from matplotlib import pyplot as plt
import numpy as np
from setup import *
import pandas as pd
from jproperties import Properties
from visual import *

configs = Properties()

# Load project details
with open('details.properties', 'rb') as config_file:
    configs.load(config_file)

# Init (Fetching and Filtering)

# database_name = configs.get('mimic4').data
# records = load_records(database_name)
# matching_records = filter_records(records, database_name)
# save_records_to_csv(matching_records)

# After Init
print("Reading matching_records.csv")
df = pd.read_csv('matching_records.csv')
# All segments array
segments = df.values

i = 1
# Extract 10 minutes of simultaneous BP and PPG signals from each record
for segment in segments:

    print(f"Segment {i} / {len(segments)}")

    segment_metadata = load_metadata_from_segment(segment[0], segment[2])

    segment_data = load_data_from_segment(round(segment_metadata.fs),
                                          0,
                                          # 600 doesn't work, incrementing issue
                                          605,
                                          segment[0],
                                          segment[2])

    # plot_wfdb_segment(segment[0], segment_data)

    abp_col, pleth_col = get_abp_pleth_col_no(segment_data)

    abp = segment_data.p_signal[:, abp_col]
    ppg = segment_data.p_signal[:, pleth_col]
    fs = segment_data.fs

    path = configs.get('path_of_data').data

    df_abp = pd.DataFrame(data=abp)
    df_abp.to_csv(f"{path}abp_{segment[0]}.csv", index=False)

    df_ppg = pd.DataFrame(data=ppg)
    df_ppg.to_csv(f"{path}ppg_{segment[0]}.csv", index=False)

    # abp_ppg_data = {'seg_name': [], 'seg_dir': [], 'abp': [], 'ppg': []}
    #
    # abp_ppg_data['seg_name'].append(segment[0])
    # abp_ppg_data['seg_dir'].append(segment[2])
    # abp_ppg_data['abp'].append(abp)
    # abp_ppg_data['ppg'].append(ppg.values)
    #
    # df_abp_ppg = pd.DataFrame(data=abp_ppg_data)
    # df_abp_ppg.to_csv('abp_ppg_data.csv', index=False)
    # break

    plot_abp_ppg(segment[0], abp, ppg, fs)
    i = i + 1
