import os
import pathlib
import shutil
import statistics
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import scipy as sp
import wfdb.processing
import pprint

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

    filenames_ppg = os.listdir('usable_ppg_fidp_data')
    filenames_ppg.sort()

    common = []

    for filename_ppg in filenames_ppg:
        sig, seg_name_ppg, end = split_filename(filename_ppg)
        for filename_bp in filenames:
            sig, seg_name_bp, end = split_filename(filename_bp)
            if sig == 'ppg':
                break
            if seg_name_ppg == seg_name_bp:
                common.append(filename_bp)
                # shutil.copy2(f"{folder}/{filename_bp}", 'usable_bp_data')

    i = 1
    count = 0

    # for filename in filenames:
    #     if "ppg" in filename:
    #         break
    #
    #     print(f"File {i}/{len(filenames)} - {filename} ")
    #
    #     df = pd.read_csv(f"{folder}/{filename}")
    #     values = df.values
    #     seg_count = 0
    #     for value in values:
    #         if value < 0 or value > 250 or np.isnan(value) or np.isinf(value):
    #             # print(value)
    #             # print(f"{filename} contains faulty values")
    #             seg_count += 1
    #
    #             # Deletion of faulty files
    #             # sig, seg_name, end = split_filename(filename)
    #             # os.remove(f"{folder}/{filename}")
    #             # os.remove(f"{folder}/ppg_{seg_name}.{end}")
    #
    #             break
    #     if seg_count > 0:
    #         print(seg_count)
    #         count += 1
    #     i += 1
    # print(count)


def process_data(fs):
    """
    Method for signal processing abp and ppg data
    :param fs: Frequency of sampling
    """
    abs_path = os.path.abspath(os.getcwd())
    bp_path = abs_path + '/usable_bp_data'
    ppg_path = abs_path + '/usable_ppg_fidp_data'
    filenames = os.listdir(bp_path)
    filenames.sort()

    i = 1
    for filename in filenames:
        sig, seg_name, end = split_filename(filename)

        print(f"Segment {i} / {len(filenames)} - {seg_name}")
        df = pd.read_csv(f"{bp_path}/abp_{seg_name}.{end}")
        values = df.values
        abp = values[:, 0]

        df = pd.read_csv(f"{ppg_path}/ppg-fidp_{seg_name}.{end}")
        values = df.values
        ppg = values[:, 0]

        # Filtering
        lpf_cutoff = 0.7  # Hz
        hpf_cutoff = 10  # Hz

        abp_filt = filter_data(lpf_cutoff, hpf_cutoff, fs, abp)
        ppg_filt = filter_data(lpf_cutoff, hpf_cutoff, fs, ppg)

        ppg_beats, alg, ppg_fidp = beat_fidp_detection(ppg_filt, fs, seg_name)
        abp_beats, alg, abp_fidp = beat_fidp_detection(abp_filt, fs, seg_name)

        # TODO
        # peaks = sp.find_peaks(ppg)
        # values = ppg[peaks]
        # new_peaks = find_highly_contrasting_values(values)
        #  -- based on indexes of contrasting values adjust peaks
        #  -- iterate for as long as needed
        # calculate_heart_rate(new_peaks)

        abp_beat_interval = len(abp_filt) / len(abp_beats)
        print(abp_beat_interval)

        ppg_beat_interval = len(ppg_filt) / len(ppg_beats)
        print(ppg_beat_interval)

        print(f"{len(abp_beats)} beats found, Heart Rate - {len(abp_beats) / (len(abp_filt)/fs) * 60}")
        print(f"{len(ppg_beats)} beats found, Heart Rate - {len(ppg_beats) / (len(ppg_filt)/fs) * 60}")

        plot_abp_ppg_with_pulse(seg_name + ' (mimic)', abp_filt, abp_beats, ppg_filt, ppg_beats, fs)

        ppg_beats, _ = sp.find_peaks(ppg_filt, distance=ppg_beat_interval/2)
        abp_beats, _ = sp.find_peaks(abp_filt, distance=abp_beat_interval/2)

        abp_beat_interval = len(abp_filt) / len(abp_beats)
        print(abp_beat_interval)

        ppg_beat_interval = len(ppg_filt) / len(ppg_beats)
        print(ppg_beat_interval)

        print(f"{len(abp_beats)} beats found, Heart Rate - {len(abp_beats) / (len(abp_filt) / fs) * 60}")
        print(f"{len(ppg_beats)} beats found, Heart Rate - {len(ppg_beats) / (len(ppg_filt) / fs) * 60}")

        plot_abp_ppg_with_pulse(seg_name + ' (sp.find_peaks)', abp_filt, abp_beats, ppg_filt, ppg_beats, fs)

        ppg_hard, ppg_soft = wfdb.processing.find_peaks(ppg_filt)
        abp_hard, abp_soft = wfdb.processing.find_peaks(abp_filt)

        # plot_abp_ppg_with_pulse(seg_name + ' (wfdb.find_peaks)', abp_filt, abp_hard, ppg_filt, ppg_hard, fs)

        abp_d1, abp_d2 = savgol_derivatives(abp_filt)
        ppg_d1, ppg_d2 = savgol_derivatives(ppg_filt)

        ppg_beats, _ = sp.find_peaks(abp_d1)
        abp_beats, _ = sp.find_peaks(ppg_d1)

        # plot_abp_ppg_with_pulse(seg_name + ' (wfdb D1)', abp_d1, abp_beats, ppg_d1, ppg_beats, fs)

        # agi, ts, val = agi_detection(ppg_fidp, fs)
        # sys, dia, tss, sysv, tsd, diav = sys_dia_detection(abp_fidp, abp_filt)
        #
        # plot_trio(seg_name, ts, val, tss, sysv, tsd, diav)
        #
        # median_agi, mean_agi = calculate_median_mean(agi, fs, 30)
        # median_sys, mean_sys = calculate_median_mean(sys, fs, 30)
        # median_dia, mean_dia = calculate_median_mean(dia, fs, 30)

        print("---")
        # Move one file at a time
        x = input("> next")

        i += 1


def process_ppg_data(path, fs):
    path = os.path.abspath(os.getcwd()) + path
    filenames = os.listdir(path)
    filenames.sort()

    median_agi = []

    i = 1
    for filename in filenames:
        sig, seg_name, end = split_filename(filename)
        print(f"File {i} / {len(filenames)} - {filename}")

        df = pd.read_csv(f"{path}ppg-fidp_{seg_name}.{end}")
        values = df.values
        ppg = values[:, 0]

        # Filtering
        lpf_cutoff = 0.7  # Hz
        hpf_cutoff = 10  # Hz

        ppg_filt = filter_data(lpf_cutoff, hpf_cutoff, fs, ppg)

        # d1, d2 = savgol_derivatives(ppg_filt)
        #
        # plot_quad(seg_name, sig, fs, ppg, ppg_filt, d1, d2)

        ppg_beats, alg, ppg_fidp = beat_fidp_detection(ppg_filt, fs, seg_name)

        agi = agi_detection(ppg_fidp, fs)

        median_agi, mean_agi = calculate_median_mean(agi, fs, 30)

        i += 1
        x = input()
    return median_agi


def process_bp_data(path, fs):
    path = os.path.abspath(os.getcwd()) + path
    filenames = os.listdir(path)
    filenames.sort()

    median_sys = []
    median_dia = []

    i = 1
    for filename in filenames:
        sig, seg_name, end = split_filename(filename)
        print(f"File {i} / {len(filenames)} - {filename}")

        df = pd.read_csv(f"{path}abp_{seg_name}.{end}")
        values = df.values
        abp = values[:, 0]

        # Filtering
        lpf_cutoff = 0.7  # Hz
        hpf_cutoff = 10  # Hz

        abp_filt = filter_data(lpf_cutoff, hpf_cutoff, fs, abp)

        # d1, d2 = savgol_derivatives(abp_filt)
        #
        # plot_quad(seg_name, sig, fs, abp, abp_filt, d1, d2)

        abp_beats, alg, abp_fidp = beat_fidp_detection(abp_filt, fs, seg_name)

        sys, dia = sys_dia_detection(abp_fidp, abp_filt)

        median_sys, mean_sys = calculate_median_mean(sys, fs, 30)

        median_dia, mean_dia = calculate_median_mean(dia, fs, 30)

        i += 1
        x = input()
    return median_sys, median_dia


def calculate_median_mean(data, fs, window):
    values = []
    median_values = []
    mean_values = []
    time_window = window
    for j in range(len(data)):
        time_passed = data[j][0] / fs
        values.append(float(data[j][1]))
        if time_passed > time_window:
            median_values.append(statistics.median(values))
            mean_values.append(statistics.mean(values))
            values = []
            time_window = time_window + window

    return median_values, mean_values


def beat_fidp_detection(data, fs, seg_name):
    # Beat and Fiducials detection from PPG
    t = len(data) / fs
    alg = 'delineator'
    try:
        beats = pulse_detection(data, 'delineator', t, 'PPG')
        fidp = fiducial_points(data, beats, fs, vis=False)
    except Exception as e:
        # print(f"Delineator error - {e}")
        alg = 'd2max'
        try:
            beats = pulse_detection(data, 'd2max', t, 'PPG')
            fidp = fiducial_points(data, beats, fs, vis=False)
        except Exception as e:
            alg = 'upslopes'
            # print(f"D2Max error - {e}")
            try:
                beats = pulse_detection(data, 'upslopes', t, 'PPG')
                fidp = fiducial_points(data, beats, fs, vis=False)
            except Exception as e:
                # print(f"Upslopes error - {e}")
                print(f"Fiducials of {seg_name} couldn't be determined - {e}")
                alg = ''
                fidp = []

    # # Create .csv files from valid data
    # if len(ppg_fidp) != 0:
    #     print("Fiducials discovered")
    #     df_ppg = pd.DataFrame(data=ppg)
    #     df_ppg.to_csv(f"{os.path.abspath(os.getcwd())}/usable_ppg_fidp_data/ppg_fidp_{seg_name}.csv", index=False)

    return beats, alg, fidp


def ct_detection(fidp, fs):
    # CT = Systolic peak - Onset
    length = min(len(fidp["pks"]), len(fidp["ons"]))
    ts = np.zeros(length, dtype=int)
    values = np.zeros(length, dtype=float)
    for beat_no in range(length):
        ts[beat_no] = fidp["pks"][beat_no]
        values[beat_no] = (fidp["pks"][beat_no] - fidp["ons"][beat_no]) / fs
    return np.column_stack((ts, values))


def agi_detection(fidp, fs):
    # (From second derivative) Aging Index = b - c - d - e
    length = min(len(fidp["bmag2d"]), len(fidp["cmag2d"]),
                 len(fidp["dmag2d"]), len(fidp["emag2d"]))
    ts = np.zeros(length, dtype=int)
    values = np.zeros(length, dtype=float)
    for beat_no in range(length):
        ts[beat_no] = fidp["a2d"][beat_no]
        values[beat_no] = (fidp["bmag2d"][beat_no] - fidp["cmag2d"][beat_no]
                           - fidp["dmag2d"][beat_no] - fidp["emag2d"][beat_no]) / fs
    return np.column_stack((ts, values)), ts, values


def sys_dia_detection(fidp, data):
    # (From filtered data) Systolic BP = pks; Diastolic BP = dia
    length = min(len(fidp["pks"]), len(fidp["dia"]))
    tss = np.zeros(length, dtype=int)
    tsd = np.zeros(length, dtype=int)
    sysv = np.zeros(length, dtype=float)
    diav = np.zeros(length, dtype=float)
    for beat_no in range(length):
        tss[beat_no] = fidp["pks"][beat_no]
        sysv[beat_no] = data[int(tss[beat_no])]
        tsd[beat_no] = fidp["dia"][beat_no]
        diav[beat_no] = data[int(tsd[beat_no])]

    # plot_extracted_data(tss, sys)
    # plot_extracted_data(tsd, dia)

    sys = np.column_stack((tss, sysv))
    dia = np.column_stack((tsd, diav))

    return sys, dia, tss, sysv, tsd, diav


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
    # if beats.any():
    #     print(f"Detected {len(beats)} beats in the {sig} signal using the {algorithm} algorithm")

    return beats


def split_filename(filename):
    x = filename.split("_", 1)
    y = x[1].split('.')
    sig = x[0]
    seg_name = y[0]
    end = y[1]
    return sig, seg_name, end


def main():
    # manual_filter_data('data')
    process_data(62.4725)
    # process_ppg_data('/usable_ppg_fidp_data/', 62.4725)
    # process_bp_data('/usable_bp_data/', 62.4725)


if __name__ == "__main__":
    main()
