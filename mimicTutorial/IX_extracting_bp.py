import sys
from pathlib import Path
import wfdb
from matplotlib import pyplot as plt
import numpy as np
from VI_ppg_differentiation import segment_names, segment_dirs
from methods import pulse_detect, fiducial_points1


def main():
    rel_segment_no = 8  # 3 and 8 are helpful
    rel_segment_name = segment_names[rel_segment_no]
    rel_segment_dir = segment_dirs[rel_segment_no]

    start_seconds = 100  # time since the start of the segment at which to begin extracting data
    no_seconds_to_load = 20
    segment_metadata = wfdb.rdheader(record_name=rel_segment_name, pn_dir=rel_segment_dir)
    print("Metadata loaded from segment: {}".format(rel_segment_name))

    fs = round(segment_metadata.fs)
    sampfrom = fs * start_seconds
    sampto = fs * (start_seconds + no_seconds_to_load)
    segment_data = wfdb.rdrecord(record_name=rel_segment_name, sampfrom=sampfrom, sampto=sampto, pn_dir=rel_segment_dir)
    print("{} seconds of data extracted from: {}".format(no_seconds_to_load, rel_segment_name))
    abp_col = []
    ppg_col = []
    for sig_no in range(0, len(segment_data.sig_name)):
        if "ABP" in segment_data.sig_name[sig_no]:
            abp_col = sig_no
    abp = segment_data.p_signal[:, abp_col]
    fs = segment_data.fs
    print("Extracted the ABP signal from column {} of the matrix of waveform data at {:.1f} Hz.".format(abp_col, fs))

    temp_fs = 125
    alg = 'd2max'
    pks = pulse_detect(abp, temp_fs, 5, alg)
    print("Detected {} beats in the ABP signal using the {} algorithm".format(len(pks), alg))

    # t = np.arange(0, (len(abp) / fs), 1.0 / fs)
    # line1 = plt.plot(t, abp, color='black', label='ABP')
    # line2 = plt.plot(t[0] + ((pks - 1) / fs), abp[pks], ".", color='red', label='peaks')
    #
    # plt.show()

    fidp = fiducial_points1(abp, pks, temp_fs, vis=False)

    pks = fidp["pks"]
    ons = fidp["ons"]
    t = np.arange(0, (len(abp) / fs), 1.0 / fs)
    plt.plot(t, abp, color='black')
    plt.plot(t[pks], abp[pks], ".", color='red')
    plt.plot(t[ons], abp[ons], ".", color='blue')

    plt.show()

    sbp = np.median(abp[fidp['pks']])
    dbp = np.median(abp[fidp['ons']])

    ons = fidp['ons']
    off = fidp['off']
    mbps = np.zeros(len(ons))
    for beat_no in range(0, len(ons)):
        mbps[beat_no] = np.mean(abp[ons[beat_no]:off[beat_no]])
    mbp = np.median(mbps)

    print('Systolic blood pressure  (SBP): {:.1f} mmHg'.format(sbp))
    print('Diastolic blood pressure (DBP): {:.1f} mmHg'.format(dbp))
    print('Mean blood pressure      (MBP): {:.1f} mmHg'.format(mbp))


if __name__ == "__main__":
    main()
