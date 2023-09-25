import sys
from pathlib import Path
import wfdb
from IV_signal_visualization import load_data_from_segment
import scipy as sp
from matplotlib import pyplot as plt
import numpy as np


def extract_ppg_data_from_segment(segment_data):
    for sig_no in range(0, len(segment_data.sig_name)):
        if "Pleth" in segment_data.sig_name[sig_no]:
            break

    ppg = segment_data.p_signal[:, sig_no]
    fs = segment_data.fs
    print(f"Extracted the PPG signal from column {sig_no} of the matrix of waveform data.")
    return ppg, fs


def filter_ppg_sos_butterworth(lpf_cutoff, hpf_cutoff, segment_data, fs):
    # Butterworth filter
    sos_ppg = sp.signal.butter(10,
                               [lpf_cutoff, hpf_cutoff],
                               btype='bp',
                               analog=False,
                               output='sos',
                               fs=segment_data.fs)

    w, h = sp.signal.sosfreqz(sos_ppg,
                              2000,
                              fs=fs)

    return sos_ppg, w, h


def filter_ppg_sos_chebyshev(lpf_cutoff, hpf_cutoff, segment_data, fs):
    # Chebyshev filter
    sos_ppg_cheb = sp.signal.cheby2(10,
                                    5,
                                    [lpf_cutoff, hpf_cutoff],
                                    btype='bp',
                                    analog=False,
                                    output='sos',
                                    fs=segment_data.fs)
    w, h = sp.signal.sosfreqz(sos_ppg_cheb,
                              2000,
                              fs=fs)

    return sos_ppg_cheb, w, h


def main():
    # The name of the MIMIC IV Waveform Database on Physionet
    database_name = 'mimic4wdb/0.1.0'

    # Segment for analysis
    segment_names = ['83404654_0005',
                     '82924339_0007']

    segment_dirs = ['mimic4wdb/0.1.0/waves/p100/p10020306/83404654',
                    'mimic4wdb/0.1.0/waves/p101/p10126957/82924339']

    rel_segment_n = 0
    rel_segment_name = segment_names[rel_segment_n]
    rel_segment_dir = segment_dirs[rel_segment_n]

    # time since the start of the segment at which to begin extracting data
    start_seconds = 20
    n_seconds_to_load = 60

    segment_metadata = wfdb.rdheader(record_name=rel_segment_name, pn_dir=rel_segment_dir)
    print(f"Metadata loaded from segment: {rel_segment_name}")
    fs = round(segment_metadata.fs)

    segment_data = load_data_from_segment(fs, start_seconds, n_seconds_to_load, rel_segment_name, rel_segment_dir)

    print(f"{n_seconds_to_load} seconds of data extracted from: {rel_segment_name}")

    ppg, fs = extract_ppg_data_from_segment(segment_data)

    # Specify cutoff in Hertz
    lpf_cutoff = 0.7
    hpf_cutoff = 10

    # sos_ppg_cheb, w, h = filter_ppg_sos_chebyshev(lpf_cutoff, hpf_cutoff, segment_data, fs)

    sos_ppg_butt, w, h = filter_ppg_sos_butterworth(lpf_cutoff, hpf_cutoff, segment_data, fs)

    # Plot filter characteristics
    fig, ax = plt.subplots()

    ax.plot(w, 20 * np.log10(np.maximum(abs(h), 1e-5)))

    ax.set_title('Butterworth bandpass filter frequency response')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [dB]')
    ax.axis((0, 20, -100, 10))
    ax.grid(which='both',
            axis='both')

    plt.show()

    ppg_filt = sp.signal.sosfiltfilt(sos_ppg_butt, ppg)

    fig, ax = plt.subplots()
    t = np.arange(0, len(ppg_filt)) / segment_data.fs

    ax.plot(t, ppg,
            linewidth=2.0,
            color='blue',
            label="original PPG")

    ax.plot(t, ppg_filt,
            linewidth=2.0,
            color='red',
            label="filtered PPG")

    ax.set(xlim=(0, n_seconds_to_load))
    ax.set_title('Butterworth filter')
    plt.xlabel('time (s)')
    plt.ylabel('PPG')
    plt.xlim([50, 60])

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
