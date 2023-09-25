import sys
from pathlib import Path
import wfdb
from pprint import pprint

# The name of the MIMIC-IV Waveform Database on PhysioNet
database_name = 'mimic4wdb/0.1.0'

segment_names = ['83404654_0005']
segment_dirs = ['mimic4wdb/0.1.0/waves/p100/p10020306/83404654']

rel_segment_no = 0
rel_segment_name = segment_names[rel_segment_no]
rel_segment_dir = segment_dirs[rel_segment_no]
print(f"Specified segment '{rel_segment_name}' in directory: '{rel_segment_dir}'")

segment_data = wfdb.rdrecord(record_name=rel_segment_name, pn_dir=rel_segment_dir)
print(f"Data loaded from segment: {rel_segment_name}")

print(f"Data stored in class of type: {type(segment_data)}")

print(f"This segment contains waveform data for the following {segment_data.n_sig} signals: {segment_data.sig_name}")
print(f"The signals are sampled at a base rate of {segment_data.fs} Hz (and some are sampled at multiples of this)")
print(f"They last for {segment_data.sig_len / (60 * segment_data.fs):.1f} minutes")

pprint(vars(segment_data))
