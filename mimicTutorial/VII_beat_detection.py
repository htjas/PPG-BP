import wfdb
from VI_ppg_differentiation import segment_names, segment_dirs
from methods import pulse_detect
import scipy as sp
from matplotlib import pyplot as plt
import numpy as np


def main():
    # 3 and 8 are helpful
    rel_segment_no = 8
    rel_segment_name = segment_names[rel_segment_no]
    rel_segment_dir = segment_dirs[rel_segment_no]

    # time since the start of the segment at which to begin extracting data
    start_seconds = 100
    no_seconds_to_load = 20

    segment_metadata = wfdb.rdheader(record_name=rel_segment_name, pn_dir=rel_segment_dir)
    print(f"Metadata loaded from segment: {rel_segment_name}")

    fs = round(segment_metadata.fs)
    samp_from = fs * start_seconds
    samp_to = fs * (start_seconds + no_seconds_to_load)

    segment_data = wfdb.rdrecord(record_name=rel_segment_name,
                                 sampfrom=samp_from,
                                 sampto=samp_to,
                                 pn_dir=rel_segment_dir)

    print(f"{no_seconds_to_load} seconds of data extracted from: {rel_segment_name}")
    abp_col = []
    ppg_col = []

    for sig_no in range(0, len(segment_data.sig_name)):
        if "ABP" in segment_data.sig_name[sig_no]:
            abp_col = sig_no
        if "Pleth" in segment_data.sig_name[sig_no]:
            ppg_col = sig_no

    abp = segment_data.p_signal[:, abp_col]
    ppg = segment_data.p_signal[:, ppg_col]
    fs = segment_data.fs

    print(f"Extracted the ABP signal from column {abp_col} of the matrix of waveform data at {fs:.1f} Hz.")
    print(f"Extracted the PPG signal from column {ppg_col} of the matrix of waveform data at {fs:.1f} Hz.")

    # filter cut-offs
    lpf_cutoff = 0.7  # Hz
    hpf_cutoff = 10  # Hz

    # Butterworth filter
    sos_filter = sp.signal.butter(10,
                                  [lpf_cutoff, hpf_cutoff],
                                  btype='bp',
                                  analog=False,
                                  output='sos',
                                  fs=fs)

    # filter PPG
    ppg_filt = sp.signal.sosfiltfilt(sos_filter, ppg)

    # Filter ABP
    abp_filt = sp.signal.sosfiltfilt(sos_filter, abp)

    temp_fs = 125

    # Pulse detection Algorithms

    ppg_ibis_d2max = pulse_detect(ppg_filt, temp_fs, 5, 'd2max')
    if ppg_ibis_d2max.any():
        print(f"Detected {len(ppg_ibis_d2max)} beats in the PPG signal using the {'D2max'} algorithm")

    ppg_ibis_upslopes = pulse_detect(ppg_filt, temp_fs, 5, 'upslopes')
    if ppg_ibis_upslopes.any():
        print(f"Detected {len(ppg_ibis_upslopes)} beats in the PPG signal using the {'Upslopes'} algorithm")

    ppg_ibis_delineator = pulse_detect(ppg_filt, temp_fs, 5, 'delineator')
    if ppg_ibis_delineator.any():
        print(f"Detected {len(ppg_ibis_delineator)} beats in the PPG signal using the {'Delineator'} algorithm")

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1,
                                        sharex=False,
                                        sharey=False,
                                        figsize=(8, 8))
    fig.suptitle('IBIs detection')

    t = np.arange(0, len(ppg_filt) / fs, 1.0 / fs)

    ax1.plot(t, ppg_filt, color='black')
    ax1.scatter(t[0] + ppg_ibis_d2max / fs,
                ppg_filt[ppg_ibis_d2max],
                color='orange',
                marker='o')
    ax1.set_ylabel('PPG [V]')
    ax1.set_title('D2Max')

    ax2.plot(t, ppg_filt, color='black')
    ax2.scatter(t[0] + ppg_ibis_upslopes / fs,
                ppg_filt[ppg_ibis_upslopes],
                color='orange',
                marker='o')
    ax2.set_ylabel('PPG [V]')
    ax2.set_title('Upslopes')

    ax3.plot(t, ppg_filt, color='black')
    ax3.scatter(t[0] + ppg_ibis_delineator / fs,
                ppg_filt[ppg_ibis_delineator],
                color='orange',
                marker='o')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('PPG [V]')
    ax3.set_title('Delineator')

    plt.show()

    temp_fs = 125
    abp_ibis_d2max = pulse_detect(abp_filt, temp_fs, 5, 'd2max')
    print("Detected {} beats in the BP signal using the {} algorithm".format(len(abp_ibis_d2max), "D2max"))

    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   sharex=False,
                                   sharey=False,
                                   figsize=(8, 8))
    fig.suptitle('IBIs detection D2Max')

    t = np.arange(0, len(ppg_filt) / fs, 1.0 / fs)

    ax1.plot(t, ppg_filt, color='black')
    ax1.scatter(t[0] + ppg_ibis_d2max / fs,
                ppg_filt[ppg_ibis_d2max],
                color='orange',
                marker='o')
    ax1.set_ylabel('PPG [au]')

    ax2.plot(t, abp_filt, color='black')
    ax2.scatter(t[0] + abp_ibis_d2max / fs,
                abp_filt[abp_ibis_d2max],
                color='orange',
                marker='o')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('ABP [au]')

    plt.show()


if __name__ == "__main__":
    main()
