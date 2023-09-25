import sys
from pathlib import Path
import wfdb
from pprint import pprint
from I_exploration import load_records
from matplotlib import pyplot as plt
import numpy as np


def load_data_from_segment(fs, start_seconds, n_seconds_to_load, rel_segment_name, rel_segment_dir):

    samp_from = fs * start_seconds
    samp_to = fs * (start_seconds + n_seconds_to_load)

    segment_data = wfdb.rdrecord(record_name=rel_segment_name,
                                 sampfrom=samp_from,
                                 sampto=samp_to,
                                 pn_dir=rel_segment_dir)

    print(f"Data loaded from segment: {rel_segment_name}")
    return segment_data


def main():
    database_name = 'mimic4wdb/0.1.0'
    # records = load_records(database_name)

    segment_names = ['83404654_0005', '82924339_0007']
    segment_dirs = ['mimic4wdb/0.1.0/waves/p100/p10020306/83404654',
                    'mimic4wdb/0.1.0/waves/p101/p10126957/82924339']

    rel_segment_no = 1
    rel_segment_name = segment_names[rel_segment_no]
    rel_segment_dir = segment_dirs[rel_segment_no]
    print(f"Specified segment '{rel_segment_name}' in directory: '{rel_segment_dir}'")

    start_seconds = 20
    n_seconds_to_load = 60

    segment_metadata = wfdb.rdheader(record_name=rel_segment_name, pn_dir=rel_segment_dir)
    print(f'Metadata loaded from segment: {rel_segment_name}')
    fs = round(segment_metadata.fs)

    segment_data = load_data_from_segment(fs, start_seconds, n_seconds_to_load, rel_segment_name, rel_segment_dir)
    print(
        f"This segment contains waveform data for the following {segment_data.n_sig} signals: {segment_data.sig_name}")
    print(f"The signals are sampled at a base rate of {segment_data.fs} Hz (and some are sampled at multiples of this)")
    print(f"They last for {segment_data.sig_len / (60 * segment_data.fs):.1f} minutes")

    pprint(vars(segment_data))

    print(f"{n_seconds_to_load} seconds of data extracted from segment {rel_segment_name}")

    title_text = f"Segment {rel_segment_name}"
    wfdb.plot_wfdb(record=segment_data,
                   title=title_text,
                   time_units='seconds')

    for sig_no in range(0, len(segment_data.sig_name)):
        if "Pleth" in segment_data.sig_name[sig_no]:
            break

    ppg = segment_data.p_signal[:, sig_no]
    fs = segment_data.fs
    print(f"Extracted the PPG signal from column {sig_no} of the matrix of waveform data.")

    t = np.arange(0, (len(ppg) / fs), 1.0 / fs)
    plt.plot(t, ppg, color='black', label='PPG')
    plt.xlim([50, 55])
    plt.show()


if __name__ == "__main__":
    main()
