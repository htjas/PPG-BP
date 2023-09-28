import sys
import numpy as np
import scipy.signal as sp
from matplotlib import pyplot as plt
import wfdb
from methods import pulse_detect, fiducial_points
from VI_ppg_differentiation import segment_names, segment_dirs


def main():
    # 3 and 8 are helpful
    rel_segment_no = 3
    rel_segment_name = segment_names[rel_segment_no]
    rel_segment_dir = segment_dirs[rel_segment_no]

    # time since the start of the segment at which to begin extracting data
    start_seconds = 100
    n_seconds_to_load = 20

    segment_metadata = wfdb.rdheader(record_name=rel_segment_name, pn_dir=rel_segment_dir)
    print(f"Metadata loaded from segment: {rel_segment_name}")

    fs = round(segment_metadata.fs)
    sampfrom = fs * start_seconds
    sampto = fs * (start_seconds + n_seconds_to_load)
    segment_data = wfdb.rdrecord(record_name=rel_segment_name,
                                 sampfrom=sampfrom,
                                 sampto=sampto,
                                 pn_dir=rel_segment_dir)

    print("{} seconds of data extracted from: {}".format(n_seconds_to_load,
                                                         rel_segment_name))

    ppg_col = []
    for sig_no in range(0, len(segment_data.sig_name)):
        if "Pleth" in segment_data.sig_name[sig_no]:
            ppg_col = sig_no

    ppg = segment_data.p_signal[:, ppg_col]
    fs = segment_data.fs

    print(f"Extracted the PPG signal from column {ppg_col} of the matrix of waveform data at {fs:.1f} Hz.")

    # filter cut-offs, hertz
    lpf_cutoff = 0.7
    hpf_cutoff = 10

    # create filter
    sos_filter = sp.butter(10, [lpf_cutoff, hpf_cutoff],
                           btype='bp',
                           analog=False,
                           output='sos',
                           fs=segment_data.fs)

    w, h = sp.sosfreqz(sos_filter, 2000, fs=fs)

    # filter PPG
    ppg_filt = sp.sosfiltfilt(sos_filter, ppg)

    # Beat detection
    temp_fs = 125
    alg = 'd2max'
    ibis = pulse_detect(ppg_filt, temp_fs, 5, alg)

    print(f"Detected {len(ibis)} beats in the PPG signal using the {alg} algorithm")

    # Plotting
    fig, (ax1) = plt.subplots(1, 1,
                              sharex=False,
                              sharey=False,
                              figsize=(8, 8))

    fig.suptitle('IBIs detection')

    t = np.arange(0, len(ppg_filt) / fs, 1.0 / fs)

    ax1.plot(t, ppg_filt, color='black')
    ax1.scatter(t[0] + ibis / fs, ppg_filt[ibis], color='orange', marker='o')
    ax1.set_ylabel('PPG [V]')
    ax1.set_title(alg)

    plt.show()

    # Fiducial points calculation and display
    fidp = fiducial_points(ppg_filt, ibis, fs, vis=True)

    plt.show()

    print("Indices of dicrotic notches:")
    print(fidp["dic"])

    delta_t = np.zeros(len(fidp["dia"]))
    for beat_no in range(len(fidp["dia"])):
        # Delta T = Dia - Sys
        delta_t[beat_no] = (fidp["dia"][beat_no] - fidp["pks"][beat_no]) / fs
    print("Values of Delta T:")
    print(delta_t)

    agi = np.zeros(len(fidp["dia"]))
    for beat_no in range(len(fidp["dia"])):
        agi[beat_no] = (fidp["bmag2d"][beat_no] - fidp["cmag2d"][beat_no]
                        - fidp["dmag2d"][beat_no] - fidp["emag2d"][beat_no]) / fs
    print("Values of Aging Index:")
    print(agi)

    ct = np.zeros(len(fidp["dia"]))
    for beat_no in range(len(fidp["dia"])):
        # CT = Systolic peak - Onset
        ct[beat_no] = (fidp["p1p"][beat_no] - fidp["ons"][beat_no]) / fs
    print("Values of CT:")
    print(ct)


if __name__ == "__main__":
    main()
