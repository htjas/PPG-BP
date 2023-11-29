import os
import pathlib
import shutil
import statistics
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import scipy as sp
from scipy.ndimage import gaussian_filter1d
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
    print(len(filenames))
    filenames.sort()

    # for filename in filenames:
    #     sig, seg_name_bp, end = split_filename(filename)
    #     if sig == 'abp':
    #         shutil.copy2(f"{folder}/{filename}", 'usable_bp_data_2')
    #     if sig == 'ppg':
    #         shutil.copy2(f"{folder}/{filename}", 'usable_ppg_data_2')
    #
    # print(len(os.listdir('usable_bp_data_2')))
    # print(len(os.listdir('usable_ppg_data_2')))

    # filenames_ppg = os.listdir('usable_ppg_fidp_data')
    # filenames_ppg.sort()
    #
    # common = []
    #
    # for filename_ppg in filenames_ppg:
    #     sig, seg_name_ppg, end = split_filename(filename_ppg)
    #     for filename_bp in filenames:
    #         sig, seg_name_bp, end = split_filename(filename_bp)
    #         if sig == 'ppg':
    #             break
    #         if seg_name_ppg == seg_name_bp:
    #             common.append(filename_bp)
    #             # shutil.copy2(f"{folder}/{filename_bp}", 'usable_bp_data')

    i = 1
    count = 0

    for filename in filenames:
        print(f"File {i}/{len(filenames)} - {filename} ")
        df = pd.read_csv(f"{folder}/{filename}")
        values = df.values
        seg_count = 0
        for value in values:
            if np.isnan(value) or np.isinf(value) or value == 0.0:
                print(value)
                print(f"{filename} contains faulty values")
                seg_count += 1

                # Deletion of faulty files
                sig, seg_name, end = split_filename(filename)
                os.remove(f"{folder}/{filename}")
                os.remove(f"usable_bp_data_2/abp_{seg_name}.{end}")

                break
        if seg_count > 0:
            print(seg_count)
            count += 1
        i += 1
    print(count)

    filenames = os.listdir(folder)
    print(len(filenames))
    filenames = os.listdir('usable_bp_data_2')
    print(len(filenames))


def process_data(fs):
    """
    Main method for signal processing abp and ppg data
    :param fs: Frequency of sampling
    """
    abs_path = os.path.abspath(os.getcwd())
    bp_path = abs_path + '/usable_bp_data_2'
    ppg_path = abs_path + '/usable_ppg_data_2'
    filenames = os.listdir(bp_path)
    filenames.sort()

    i = 1
    for filename in filenames:
        # if i != 14:
        #     i += 1
        #     continue
        # Data Reading
        seg_name, abp, ppg = read_seg_data(i, len(filenames), filename, bp_path, ppg_path, fs)

        # Data Pre-Processing
        abp_filt, ppg_filt = pre_process_data(abp, ppg, fs, seg_name)

        # Signal Processing (Beat and Dip detection)
        abp_beats, ppg_beats, abp_dips, ppg_dips = signal_processing(seg_name, abp_filt, ppg_filt, fs)
        delay = 18  # = 288 ms

        abp_fidp = fiducial_points(abp_filt, abp_beats, fs, vis=False, header='ABP of ' + seg_name)
        ppg_fidp = fiducial_points(ppg_filt, ppg_beats, fs, vis=False, header='PPG of ' + seg_name)

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
        # x = input("> next")

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

        ppg_filt = filter_butterworth(lpf_cutoff, hpf_cutoff, fs, ppg)

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

        abp_filt = filter_butterworth(lpf_cutoff, hpf_cutoff, fs, abp)

        # d1, d2 = savgol_derivatives(abp_filt)
        #
        # plot_quad(seg_name, sig, fs, abp, abp_filt, d1, d2)

        abp_beats, alg, abp_fidp = beat_fidp_detection(abp_filt, fs, seg_name)

        sys, dia, tss, sysv, tsd, diav = sys_dia_detection(abp_fidp, abp_filt)

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
    beats = []
    try:
        beats = pulse_detection(data, 'delineator', t, 'PPG')
        fidp = fiducial_points(data, beats, fs, vis=False, header=seg_name)
    except Exception as e:
        # print(f"Delineator error - {e}")
        alg = 'd2max'
        try:
            beats = pulse_detection(data, 'd2max', t, 'PPG')
            fidp = fiducial_points(data, beats, fs, vis=False, header=seg_name)
        except Exception as e:
            alg = 'upslopes'
            # print(f"D2Max error - {e}")
            try:
                beats = pulse_detection(data, 'upslopes', t, 'PPG')
                fidp = fiducial_points(data, beats, fs, vis=False, header=seg_name)
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


def read_seg_data(i, i_len, filename, bp_path, ppg_path, fs):
    sig, seg_name, end = split_filename(filename)
    print(f"Segment {i} / {i_len} - {seg_name}")
    df = pd.read_csv(f"{bp_path}/abp_{seg_name}.{end}")
    values = df.values
    abp = values[:, 0]

    df = pd.read_csv(f"{ppg_path}/ppg_{seg_name}.{end}")
    values = df.values
    ppg = values[:, 0]
    # plot_abp_ppg(seg_name, abp, ppg, fs)

    return seg_name, abp, ppg


def pre_process_data(abp, ppg, fs, seg_name):
    lpf_cutoff = 0.7  # Hz
    hpf_cutoff = 10  # Hz

    # Butterworth filter
    # abp = filter_butterworth(abp, lpf_cutoff, hpf_cutoff, fs)
    # ppg = filter_butterworth(ppg, lpf_cutoff, hpf_cutoff, fs)
    # plot_abp_ppg(seg_name + ' butt filtered', abp, ppg, fs)

    # Chebyshev filter
    # abp = filter_chebyshev(abp, lpf_cutoff, hpf_cutoff, fs)
    # ppg = filter_chebyshev(ppg, lpf_cutoff, hpf_cutoff, fs)
    # plot_abp_ppg(seg_name + ' cheb filtered', abp, ppg, fs)

    # Whiskers filter
    abp = whiskers_filter(abp)
    ppg = whiskers_filter(ppg)
    # plot_abp_ppg(seg_name + ' whiskers filtered', abp, ppg, fs)

    # Gaussian filter
    abp = gaussian_filter1d(abp, 2)
    ppg = gaussian_filter1d(ppg, 2)
    # plot_abp_ppg(seg_name + ' gauss smooth', abp, ppg, fs)

    return abp, ppg


def filter_butterworth(data, lpf_cutoff, hpf_cutoff, fs):
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
    sos = sos_butt, w, h
    return sp.sosfiltfilt(sos[0], data)


def filter_chebyshev(data, lpf_cutoff, hpf_cutoff, fs):
    # Chebyshev filter
    sos_cheb = sp.cheby2(10,
                         5,
                         [lpf_cutoff, hpf_cutoff],
                         btype='bp',
                         analog=False,
                         output='sos',
                         fs=fs)
    w, h = sp.sosfreqz(sos_cheb,
                       2000,
                       fs=fs)
    return sp.sosfiltfilt(sos_cheb[0], data)


def whiskers_filter(data):
    bp = plt.boxplot(data)
    plt.close()

    # get lower and upper amplitude thresholds
    whiskers = [whiskers.get_ydata() for whiskers in bp['whiskers']]
    lower_amp = whiskers[0][1]
    upper_amp = whiskers[1][1]

    # get all indexes of outliers, outside the amplitude thresholds
    ind_outliers = []
    for i in range(len(data)):
        if data[i] < lower_amp or data[i] > upper_amp:
            ind_outliers.append(i)

    if len(ind_outliers) != 0:
        # get all grouped arrays with values not within the whiskers range
        ind_consecutives = []
        current_group = [ind_outliers[0]]
        for i in range(1, len(ind_outliers)):
            if ind_outliers[i] == ind_outliers[i - 1] + 1:
                current_group.append(ind_outliers[i])
            else:
                ind_consecutives.append(current_group)
                current_group = [ind_outliers[i]]
        ind_consecutives.append(current_group)

        for array_indexes in ind_consecutives:
            array_values = data[array_indexes]
            # get top (min or max) value of each consecutive group
            if array_values[0] < lower_amp:
                top_val = min(array_values)
                top_ind = array_indexes[np.argmin(array_values)]
                coef = top_val / lower_amp
            else:
                top_val = max(array_values)
                top_ind = array_indexes[np.argmax(array_values)]
                coef = top_val / upper_amp
            # calculate and (not) assign the top value according to attenuating coefficient
            adj_top_val = top_val / coef * 1.01
            if lower_amp > top_val > adj_top_val:
                continue
            if upper_amp < top_val < adj_top_val:
                continue
            data[top_ind] = adj_top_val
            # create function for calculation of other new values
            if 0 < array_indexes[0] and array_indexes[-1] < len(data)-1:
                one_minus_threshold_val = data[array_indexes[0] - 1]
                one_minus_threshold_ind = array_indexes[0] - 1
                one_plus_threshold_val = data[array_indexes[-1] + 1]
                one_plus_threshold_ind = array_indexes[-1] + 1
            elif array_indexes[0] == 0:
                one_minus_threshold_val = data[array_indexes[0]]
                one_minus_threshold_ind = array_indexes[0]
            else:  # array_indexes[-1] == len(data)-1:
                one_plus_threshold_val = data[array_indexes[-1]]
                one_plus_threshold_ind = array_indexes[-1]
            distance_to_top = top_ind - one_minus_threshold_ind
            distance_to_bottom = one_plus_threshold_ind - top_ind
            f1 = (adj_top_val - one_minus_threshold_val) / distance_to_top
            f2 = (one_plus_threshold_val - adj_top_val) / distance_to_bottom
            # assign new values
            x1, x2 = 1, 1
            for ind in array_indexes:
                if ind < top_ind:
                    data[ind] = (f1 * x1) + one_minus_threshold_val
                    x1 += 1
                elif ind > top_ind:
                    data[ind] = (f2 * x2) + adj_top_val
                    x2 += 1

    return data


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


def signal_processing(seg_name, abp, ppg, fs):
    # First iteration of beat finding (MIMIC default methods)
    abp_beats, ppg_beats = get_optimal_beats_lists(abp, ppg, fs)
    abp_beat_interval = len(abp) / len(abp_beats)
    ppg_beat_interval = len(ppg) / len(ppg_beats)
    # plot_abp_ppg_with_pulse(seg_name + ' (mimic)', abp, abp_beats, ppg, ppg_beats, fs)

    # Second iteration of beat finding (SP manual methods)
    abp_beats, _ = sp.find_peaks(abp, distance=abp_beat_interval * .75, prominence=10)
    ppg_beats, _ = sp.find_peaks(ppg, distance=ppg_beat_interval * .75, prominence=0.000)
    # plot_abp_ppg_with_pulse(seg_name + ' PEAKS (sp.find_peaks)', abp, abp_beats, ppg, ppg_beats, fs)
    print(f"ABP beats - {len(abp_beats)}, Heart Rate - {len(abp_beats) / (len(abp) / fs) * 60}")
    print(f"PPG beats - {len(ppg_beats)}, Heart Rate - {len(ppg_beats) / (len(ppg) / fs) * 60}")

    abp_dips, _ = sp.find_peaks(-abp, distance=abp_beat_interval * .75, prominence=10)
    ppg_dips, _ = sp.find_peaks(-ppg, distance=ppg_beat_interval * .75, prominence=0.001)
    # plot_abp_ppg_with_pulse(seg_name + ' ONSETS (sp.find_peaks)', abp, abp_dips, ppg, ppg_dips, fs)

    return abp_beats, ppg_beats, abp_dips, ppg_dips


def pulse_detection(data, algorithm, duration, sig):
    # Pulse detection Algorithms
    temp_fs = 125

    beats = pulse_detect(data, temp_fs, 5, algorithm, duration)
    # if beats.any():
    #     print(f"Detected {len(beats)} beats in the {sig} signal using the {algorithm} algorithm")

    return beats


def get_optimal_beats_lists(abp, ppg, fs):
    """
    Signal processing script for examining which default beat detection algorithm is the most efficient
    :param abp: List of ABP data
    :param ppg: List of PPG data
    :param fs: Sampling frequency
    :return: Lists of optimally detected ABP and PPG pulses
    """
    abp_beats1 = pulse_detection(abp, 'd2max', len(abp) / fs, 'abp')
    ppg_beats1 = pulse_detection(ppg, 'd2max', len(ppg) / fs, 'ppg')
    abp_beats2 = pulse_detection(abp, 'upslopes', len(abp) / fs, 'abp')
    ppg_beats2 = pulse_detection(ppg, 'upslopes', len(ppg) / fs, 'ppg')
    abp_beats3 = pulse_detection(abp, 'delineator', len(abp) / fs, 'abp')
    ppg_beats3 = pulse_detection(ppg, 'delineator', len(ppg) / fs, 'ppg')

    avg1 = (len(abp_beats1) + len(ppg_beats1)) / 2
    avg2 = (len(abp_beats2) + len(ppg_beats2)) / 2
    avg3 = (len(abp_beats3) + len(ppg_beats3)) / 2

    sorted_avg = sorted([avg1, avg2, avg3])
    diff_avg12 = sorted_avg[1] - sorted_avg[0]
    diff_avg23 = sorted_avg[2] - sorted_avg[1]

    outlier = None
    if diff_avg23 > diff_avg12 * 2:
        outlier = sorted_avg[2]
    elif diff_avg12 > diff_avg23 * 2:
        outlier = sorted_avg[0]

    diff1 = abs(len(abp_beats1) - len(ppg_beats1))
    diff2 = abs(len(abp_beats2) - len(ppg_beats2))
    diff3 = abs(len(abp_beats3) - len(ppg_beats3))

    diffper1 = diff1 / avg1 * 100
    diffper2 = diff2 / avg2 * 100
    diffper3 = diff3 / avg3 * 100

    if outlier is None:
        abp_opt, ppg_opt = get_best_beats_from_diffper(diffper1, diffper2, diffper3,
                                                       abp_beats1, abp_beats2, abp_beats3,
                                                       ppg_beats1, ppg_beats2, ppg_beats3)
    else:
        if outlier == avg1:
            diffper1 = 100
        elif outlier == avg2:
            diffper2 = 100
        elif outlier == avg3:
            diffper3 = 100
        abp_opt, ppg_opt = get_best_beats_from_diffper(diffper1, diffper2, diffper3,
                                                       abp_beats1, abp_beats2, abp_beats3,
                                                       ppg_beats1, ppg_beats2, ppg_beats3)

    return abp_opt, ppg_opt


def get_best_beats_from_diffper(diffper1, diffper2, diffper3,
                                abp_beats1, abp_beats2, abp_beats3,
                                ppg_beats1, ppg_beats2, ppg_beats3):
    smallest_diffper = min(diffper1, diffper2, diffper3)

    abp_opt = []
    ppg_opt = []

    if smallest_diffper == diffper1:
        abp_opt = abp_beats1
        ppg_opt = ppg_beats1
    elif smallest_diffper == diffper2:
        abp_opt = abp_beats2
        ppg_opt = ppg_beats2
    elif smallest_diffper == diffper3:
        abp_opt = abp_beats3
        ppg_opt = ppg_beats3

    return abp_opt, ppg_opt


def split_filename(filename):
    x = filename.split("_", 1)
    y = x[1].split('.')
    sig = x[0]
    seg_name = y[0]
    end = y[1]
    return sig, seg_name, end


def main():
    # manual_filter_data('usable_ppg_data_2')
    process_data(62.4725)
    # process_ppg_data('/usable_ppg_fidp_data/', 62.4725)
    # process_bp_data('/usable_bp_data/', 62.4725)


if __name__ == "__main__":
    main()
