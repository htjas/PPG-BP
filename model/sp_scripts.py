import os
import pathlib
import pandas as pd
import scipy as sp
from visual import *
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
        for value in values:
            if value < 0 or value > 250:
                print(value)
                print(f"{filename} contains faulty values")
                count += 1

                # sig, seg_name, end = split_filename(filename)
                # os.remove(f"{folder}/{filename}")
                # os.remove(f"{folder}/ppg_{seg_name}.{end}")

                break
        print(count)

        i += 1


def process_data(folder, fs):
    """
    Method for signal processing abp and bp data
    :param fs:
    :param folder: name of folder containing abp and ppg data
    """
    filenames = os.listdir(folder)
    filenames.sort()

    i = 1
    for filename in filenames:
        if "ppg" in filename:
            break

        sig, seg_name, end = split_filename(filename)

        print(f"File {i} / {round(len(filenames) / 2)} - {filename}")
        df = pd.read_csv(f"{folder}/abp_{seg_name}.{end}")
        abp = df.to_numpy()
        # print(f"abp signal of segment - {seg_name}, tenth value - {abp[10]}")

        df = pd.read_csv(f"{folder}/ppg_{seg_name}.{end}")
        ppg = df.to_numpy()
        # print(f"ppg signal of segment - {seg_name}, tenth value - {ppg[10]}")

        # Raw plot
        plot_abp_ppg(seg_name, abp, ppg, fs)

        # filter cut-offs
        lpf_cutoff = 0.7  # Hz
        hpf_cutoff = 10  # Hz

        # Filtered Plot
        plot_abp_ppg(seg_name + " (filtered)",
                     abp,
                     filter_ppg(lpf_cutoff, hpf_cutoff, fs, ppg),
                     fs)

        # Move one file at a time
        x = input()

        i += 1


def filter_ppg(lpf, hpf, fs, ppg):
    sos = filter_butterworth(lpf, hpf, fs)
    # sos = filter_ppg_sos_chebyshev(lpf, hpf, fs)
    ppg_filtered = sp.signal.sosfiltfilt(sos[0], ppg, 0)
    return ppg_filtered


def filter_butterworth(lpf_cutoff, hpf_cutoff, fs):
    # Butterworth filter
    sos_butt = sp.signal.butter(10,
                                [lpf_cutoff, hpf_cutoff],
                                btype='bp',
                                analog=False,
                                output='sos',
                                fs=fs)

    w, h = sp.signal.sosfreqz(sos_butt,
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


def split_filename(filename):
    x = filename.split("_", 1)
    y = x[1].split('.')
    sig = x[0]
    seg_name = y[0]
    end = y[1]
    return sig, seg_name, end


def main():
    process_data('data', 62.4725)
    # manual_filter_data('data')


if __name__ == "__main__":
    main()
