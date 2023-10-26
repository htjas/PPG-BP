import os
import pathlib
import pandas as pd
import scipy as sp
from visual import *
from methods import *
from init_scripts import *


def manual_filter_data(folder):
    """
    Method for removing the data that contains extreme values (possibly faulty)
    :param folder: the path where the files are stored
    """
    filenames = os.listdir(folder)
    filenames.sort()

    i = 1
    count = 0
    for filename in filenames:
        if "ppg" in filename:
            break

        df = pd.read_csv(f"{folder}/{filename}")
        values = df.values
        print(f"File {i}/{len(filenames)} - {filename} ")
        seg_count = 0
        for value in values:
            if value < 0 or value > 250 or np.isnan(value) or np.isinf(value):
                # print(value)
                # print(f"{filename} contains faulty values")
                seg_count += 1

                # Deletion of faulty files
                # sig, seg_name, end = split_filename(filename)
                # os.remove(f"{folder}/{filename}")
                # os.remove(f"{folder}/ppg_{seg_name}.{end}")

                break
        if seg_count > 0:
            print(seg_count)
            count += 1
        i += 1
    print(count)


def process_data(path, fs):
    """
    Method for signal processing abp and bp data
    :param fs: Frequency of sampling
    :param path: name of folder containing abp and ppg data
    """
    filenames = os.listdir(path)
    filenames.sort()

    i = 1
    for filename in filenames:
        if "ppg" in filename:
            break

        sig, seg_name, end = split_filename(filename)

        print(f"File {i} / {round(len(filenames) / 2)} - {filename}")
        df = pd.read_csv(f"{path}abp_{seg_name}.{end}")
        values = df.values
        abp = values[:, 0]

        df = pd.read_csv(f"{path}ppg_{seg_name}.{end}")
        values = df.values
        ppg = values[:, 0]

        # # Raw plot
        # plot_abp_ppg(seg_name, abp, ppg, fs)

        # Filtering
        lpf_cutoff = 0.7  # Hz
        hpf_cutoff = 10  # Hz

        ppg_filt = filter_data(lpf_cutoff, hpf_cutoff, fs, ppg)
        abp_filt = filter_data(lpf_cutoff, hpf_cutoff, fs, abp)

        # # Filtered and Derivative plot
        # plot_abp_ppg(seg_name + " (filtered)",
        #              abp_filt,
        #              ppg_filt,
        #              fs)
        #
        # # Savitzky-Golay Derivation
        # d1, d2 = savgol_derivatives(ppg_filt)
        #
        # plot_abp_ppg(seg_name + " (PPG D1)",
        #              abp_filt,
        #              d1,
        #              fs)
        #
        # plot_abp_ppg(seg_name + " (PPG D2)",
        #              abp_filt,
        #              d2,
        #              fs)

        # Beat detection
        t = len(ppg_filt) / fs
        try:
            ppg_beats = pulse_detection(ppg_filt, 'delineator', t, 'PPG')
            abp_beats = pulse_detection(abp_filt, 'delineator', t, 'ABP')
        except Exception as e:
            print(f"{filename} is faulty - {e}")
            i += 1
            continue

        # # Move one file at a time
        print("---")
        x = input()

        i += 1


def filter_data(lpf, hpf, fs, data):
    sos = filter_butterworth(lpf, hpf, fs)
    # sos = filter_ppg_sos_chebyshev(lpf, hpf, fs)
    data_filtered = sp.sosfiltfilt(sos[0], data)
    return data_filtered


def filter_butterworth(lpf_cutoff, hpf_cutoff, fs):
    # Butterworth filter
    sos_butt = sp.butter(10,
                         [lpf_cutoff, hpf_cutoff],
                         btype='bp',
                         analog=False,
                         output='sos',
                         fs=fs)

    w, h = sp.sosfreqz(sos_butt,
                       2000,
                       fs=fs)

    return sos_butt, w, h


def filter_ppg_sos_chebyshev(lpf_cutoff, hpf_cutoff, fs):
    # Chebyshev filter
    sos_cheb = sp.signal.cheby2(10,
                                5,
                                [lpf_cutoff, hpf_cutoff],
                                btype='bp',
                                analog=False,
                                output='sos',
                                fs=fs)
    w, h = sp.signal.sosfreqz(sos_cheb,
                              2000,
                              fs=fs)

    return sos_cheb, w, h


def savgol_derivatives(ppg_filt):
    # Calculate first derivative
    d1ppg = sp.savgol_filter(ppg_filt,
                             9,
                             5,
                             deriv=1)
    # Calculate second derivative
    d2ppg = sp.savgol_filter(ppg_filt,
                             9,
                             5,
                             deriv=2)
    return d1ppg, d2ppg


def pulse_detection(data, algorithm, duration, sig):
    # Pulse detection Algorithms
    temp_fs = 125

    beats = pulse_detect(data, temp_fs, 5, algorithm, duration)
    if beats.any():
        print(f"Detected {len(beats)} beats in the {sig} signal using the {algorithm} algorithm")

    return beats


def split_filename(filename):
    x = filename.split("_", 1)
    y = x[1].split('.')
    sig = x[0]
    seg_name = y[0]
    end = y[1]
    return sig, seg_name, end


def main():
    process_data('data/', 62.4725)
    # manual_filter_data('data')


if __name__ == "__main__":
    main()
