import inspect
import os
import pathlib
import shutil
import statistics
import matplotlib.pyplot as plt
import numpy as np
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

    # df = pd.read_csv(os.path.abspath(os.getcwd()) + '/faulty_data/segments_low_hr.csv')
    # low_hrs = df.values
    df = pd.read_csv(os.path.abspath(os.getcwd()) + '/faulty_data/segments_big_diff.csv')
    big_diffs = df.values

    for filename in filenames:
        print(f"File {i}/{len(filenames)} - {filename} ")
        # for lh in low_hrs:
        #     if lh[0] in filename:
        #         print(lh[0], filename)
        #         sig, seg_name, end = split_filename(filename)
        #         os.remove(f"{folder}/{filename}")
        #         os.remove(f"usable_bp_data_2/abp_{seg_name}.{end}")
        for bd in big_diffs:
            if bd[0] in filename:
                print(bd[0], filename)
                sig, seg_name, end = split_filename(filename)
                try:
                    os.remove(f"{folder}/{filename}")
                    os.remove(f"usable_bp_data_2/abp_{seg_name}.{end}")
                except Exception as e:
                    print(e)
        # df = pd.read_csv(f"{folder}/{filename}")
        # values = df.values
        # seg_count = 0
        # for value in values:
        #     if np.isnan(value) or np.isinf(value) or value == 0.0:
        #         print(value)
        #         print(f"{filename} contains faulty values")
        #         seg_count += 1
        #
        #         # Deletion of faulty files
        #         # sig, seg_name, end = split_filename(filename)
        #         # os.remove(f"{folder}/{filename}")
        #         # os.remove(f"usable_bp_data_2/abp_{seg_name}.{end}")
        #
        #         break
        # if seg_count > 0:
        #     print(seg_count)
        #     count += 1
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
    tot_ppg_sys, tot_ppg_dia, tot_abp_sys, tot_abp_dia = np.array([]), np.array([]), np.array([]), np.array([])
    for filename in filenames:
        a_sys, p_sys, a_dia, p_dia = np.array([]), np.array([]), np.array([]), np.array([])
        if i != 11:
            i += 1
            continue
        try:
            # 1: Data Reading
            seg_name, abp, ppg = read_seg_data(i, len(filenames), filename, bp_path, ppg_path, fs)

            # 2: Data Pre-Processing
            abp_filt, ppg_filt = pre_process_data(abp, ppg, fs, seg_name)

            # 3: Signal Processing (Detecting Beats and Fiducials)
            result = signal_processing(seg_name, abp_filt, ppg_filt, fs)

            abp_fidp = fiducial_points(result['abp'], result['abp_beats'], fs, vis=False, header='ABP of ' + seg_name)
            ppg_fidp = fiducial_points(result['ppg'], result['ppg_beats'], fs, vis=False, header='PPG of ' + seg_name)

            # 4: Feature extraction and grouping
            a_sys, p_sys, a_dia, p_dia = extract_features(abp_fidp, ppg_fidp, result, fs)
            print(len(a_sys), len(p_sys), len(a_dia), len(p_dia))

        except Exception as e:
            print('ERROR', e)

        tot_ppg_sys = np.concatenate((tot_ppg_sys, p_sys))
        tot_ppg_dia = np.concatenate((tot_ppg_dia, p_dia))
        tot_abp_sys = np.concatenate((tot_abp_sys, a_sys))
        tot_abp_dia = np.concatenate((tot_abp_dia, a_dia))

        print("---")
        # Move one file at a time
        # x = input("> next")
        i += 1

    # Save extracted features to .csv
    save_split_features([[tot_ppg_sys, 'tot_ppg_sys'],
                         [tot_ppg_dia, 'tot_ppg_dia'],
                         [tot_abp_sys, 'tot_abp_sys'],
                         [tot_abp_dia, 'tot_abp_dia']])


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


def extract_features(abp_fidp, ppg_fidp, result, fs):
    a_sys, a_dia, a_tss, a_sysv, a_tsd, a_diav = sys_dia_detection(abp_fidp, result['abp'])
    p_sys, p_dia, p_tss, p_sysv, p_tsd, p_diav = sys_dia_detection(ppg_fidp, result['ppg'])

    a_ts_i, p_ts_i = group_timestamps(a_tss, p_tss, a_tsd, p_tsd)

    if (len(a_ts_i) != 0 or len(p_ts_i)) != 0 and len(a_ts_i) == len(p_ts_i):
        a_sys, a_sysv = a_sys[a_ts_i], a_sysv[a_ts_i]
        p_sys, p_sysv = p_sys[p_ts_i], p_sysv[p_ts_i]
        a_dia, a_diav = a_dia[a_ts_i], a_diav[a_ts_i]
        p_dia, p_diav = p_dia[p_ts_i], p_diav[p_ts_i]
    else:
        raise Exception('No matching timestamps found')

    # median_ct, mean_ct = calculate_median_mean(ct, fs, 30)
    # abp_median_sys, abp_mean_sys = calculate_median_mean(a_sys, fs, 30)
    # abp_median_dia, abp_mean_dia = calculate_median_mean(a_dia, fs, 30)
    #
    # ppg_median_sys, ppg_mean_sys = calculate_median_mean(p_sys, fs, 30)
    # ppg_median_dia, ppg_mean_dia = calculate_median_mean(p_dia, fs, 30)

    # ppg_fdf = frequency_domain_features(ppg_filt, fs)
    # print(ppg_fdf)
    # ct, tsc, ctv = ct_detection(ppg_fidp, fs)

    return a_sysv, p_sysv, a_diav, p_diav


def group_timestamps(a_tss, p_tss, a_tsd, p_tsd):

    abp_sys_timestamps, ppg_sys_timestamps = group_a_b(a_tss, p_tss)
    abp_dia_timestamps, ppg_dia_timestamps = group_a_b(a_tsd, p_tsd)

    a_comm = np.intersect1d(abp_sys_timestamps, abp_dia_timestamps)
    p_comm = np.intersect1d(ppg_sys_timestamps, ppg_dia_timestamps)

    return a_comm, p_comm


def group_a_b(a_ts, b_ts):
    a_ts_i, b_ts_i = [], []
    i, as_ext, bs_ext = 0, 0, 0
    while i < min(len(a_ts), len(b_ts)):
        a = a_ts[i]
        b = b_ts[i]
        if i < len(a_ts) - 1 and i < len(b_ts) - 1:
            if a <= b <= a_ts[i + 1] - 5 and b - 20 <= a <= b + 20:
                a_ts_i.append(i + as_ext)
                b_ts_i.append(i + bs_ext)
                i += 1
            elif b <= a <= b_ts[i + 1] - 5 and a - 20 <= b <= a + 20:
                a_ts_i.append(i + as_ext)
                b_ts_i.append(i + bs_ext)
                i += 1
            else:
                if a < b:
                    a_ts = a_ts[a_ts != a]
                    as_ext += 1
                elif a > b:
                    b_ts = b_ts[b_ts != b]
                    bs_ext += 1
        else:
            break

    a_ts_i, b_ts_i = equal_out_by_shortening(a_ts_i, b_ts_i)
    return a_ts_i, b_ts_i


def save_split_features(features):
    for feat in features:
        df = pd.DataFrame(data=feat[0])
        df.to_csv(f"features/all/{feat[1]}.csv", index=False)

    # for feat in features:
    #     mid = int(len(feat[0]) / 2)
    #     df = pd.DataFrame(data=feat[0][:mid])
    #     df.to_csv(f"features/training/{feat[1]}_train.csv", index=False)
    #     df = pd.DataFrame(data=feat[0][mid:])
    #     df.to_csv(f"features/testing/{feat[1]}_test.csv", index=False)

    # mid = int(len(tot_median_sys) / 2)
    # df = pd.DataFrame(data=tot_median_sys[:mid])
    # df.to_csv('features/training/total_median_systoles_train.csv', index=False)
    # df = pd.DataFrame(data=tot_median_sys[mid:])
    # df.to_csv('features/testing/total_median_systoles_test.csv', index=False)
    #
    # mid = int(len(tot_median_dia) / 2)
    # df = pd.DataFrame(data=tot_median_dia[:mid])
    # df.to_csv('features/training/total_median_diastoles_train.csv', index=False)
    # df = pd.DataFrame(data=tot_median_dia[mid:])
    # df.to_csv('features/testing/total_median_diastoles_test.csv', index=False)


def equal_out_by_shortening(a_ts, p_ts):
    if len(a_ts) > len(p_ts):
        diff = len(a_ts) - len(p_ts)
        a_ts = a_ts[:-diff]
    elif len(a_ts) < len(p_ts):
        diff = len(p_ts) - len(a_ts)
        p_ts = p_ts[:-diff]
    return a_ts, p_ts


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

    return np.array(median_values), np.array(mean_values)


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
        ts[beat_no] = beat_no  # fidp["pks"][beat_no]
        values[beat_no] = (fidp["pks"][beat_no] - fidp["ons"][beat_no]) / fs
    return np.column_stack((ts, values)), ts, values


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
    sys = fidp['pks']
    dia = fidp['dia']
    sys, dia = group_sys_dia(sys, dia)
    length = min(len(sys), len(dia))
    tss = np.zeros(length, dtype=int)
    tsd = np.zeros(length, dtype=int)
    sysv = np.zeros(length, dtype=float)
    diav = np.zeros(length, dtype=float)
    beat_no = 0
    while beat_no < len(tss):
        sys_beat, dia_beat = data[sys[beat_no]], data[dia[beat_no]]
        if sys_beat >= dia_beat:
            tss[beat_no] = sys[beat_no]
            sysv[beat_no] = data[sys[beat_no]]
            tsd[beat_no] = dia[beat_no]
            diav[beat_no] = data[dia[beat_no]]
            beat_no += 1
        else:
            tss = np.delete(tss, beat_no)
            sysv = np.delete(sysv, beat_no)
            tsd = np.delete(tsd, beat_no)
            diav = np.delete(diav, beat_no)
            beat_no += 1

    # plot_extracted_data('SYS + DIA', sysv, diav)

    sys = np.column_stack((tss, sysv))
    dia = np.column_stack((tsd, diav))

    return sys, dia, tss, sysv, tsd, diav


def group_sys_dia(sys, dia):
    i = 0
    while i < min(len(sys), len(dia)):
        s = sys[i]
        d = dia[i]
        if i < len(sys) - 1:
            if s < d < sys[i + 1]:
                i += 1
            else:
                if s < d:
                    sys = sys[sys != s]
                elif s > d:
                    dia = dia[dia != d]
        else:
            break
    sys, dia = equal_out_by_shortening(sys, dia)
    return sys, dia


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
    # 1st Gaussian filter
    abp = gaussian_filter1d(abp, sigma=2)
    ppg = gaussian_filter1d(ppg, sigma=2)
    # plot_abp_ppg(seg_name + ' gauss smooth', abp, ppg, fs)

    # lpf_cutoff = 0.2  # Hz
    # hpf_cutoff = 2  # Hz
    #
    # # Butterworth filter
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

    # 2nd Gaussian filter
    med_a = np.median(abp)
    med_p = np.median(ppg)
    std_a = np.std(abp)
    std_p = np.std(ppg)
    sigma_a = med_a / std_a * 0.5
    sigma_p = med_p / std_p * 0.3
    # print(f"Sigma ABP - {sigma_a}, Sigma PPG - {sigma_p}")
    abp = gaussian_filter1d(abp, sigma=sigma_a)
    ppg = gaussian_filter1d(ppg, sigma=sigma_p)
    # plot_abp_ppg(seg_name + ' gauss smooth', abp, ppg, fs)

    # Standardization
    # mean_a = np.mean(abp)
    # mean_p = np.mean(ppg)
    # stand_abp = (abp - mean_a) / std_a
    # stand_ppg = (ppg - mean_p) / std_p
    #
    # plot_abp_ppg(seg_name + ' Standardized', stand_abp, stand_ppg, fs)

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
            # check if first group index is also first overall index
            if array_indexes[0] == 0:
                one_minus_threshold_val = data[array_indexes[0]]
                one_minus_threshold_ind = array_indexes[0]
            else:
                one_minus_threshold_val = data[array_indexes[0] - 1]
                one_minus_threshold_ind = array_indexes[0] - 1
            # check if last group index is also last overall index
            if array_indexes[-1] == len(data) - 1:
                one_plus_threshold_val = data[array_indexes[-1]]
                one_plus_threshold_ind = array_indexes[-1]
            else:
                one_plus_threshold_val = data[array_indexes[-1] + 1]
                one_plus_threshold_ind = array_indexes[-1] + 1
            # create function for calculation of other new values
            distance_to_top = top_ind - one_minus_threshold_ind
            if distance_to_top == 0:
                distance_to_top = 1
            distance_to_bottom = one_plus_threshold_ind - top_ind
            if distance_to_bottom == 0:
                distance_to_bottom = 1
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


def frequency_domain_features(ppg, sampling_rate):
    # Compute the Fast Fourier Transform (FFT)
    fft_result = np.fft.fft(ppg)

    # Compute the frequencies corresponding to the FFT result
    frequencies = np.fft.fftfreq(len(fft_result), 1 / sampling_rate)

    # Only consider positive frequencies (since PPG is a real-valued signal)
    positive_frequencies = frequencies[:len(frequencies) // 2]

    # Magnitude spectrum (absolute values of FFT result)
    magnitude_spectrum = np.abs(fft_result[:len(fft_result) // 2])

    # Find the index corresponding to the maximum magnitude
    peak_frequency_index = np.argmax(magnitude_spectrum)
    peak_frequency = positive_frequencies[peak_frequency_index]

    # Other frequency domain features
    mean_frequency = np.sum(positive_frequencies * magnitude_spectrum) / np.sum(magnitude_spectrum)
    total_power = np.sum(magnitude_spectrum)
    normalized_power_at_peak = magnitude_spectrum[peak_frequency_index] / total_power

    return {
        'peak_frequency': peak_frequency,
        'mean_frequency': mean_frequency,
        'total_power': total_power,
        'normalized_power_at_peak': normalized_power_at_peak
    }


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
    # plot_abp_ppg_with_pulse(seg_name + ' (mimic)', abp, abp_beats, ppg, ppg_beats, fs)

    # For comparison: beat detection from mean crossing
    beats_a, beats_p = get_beats_from_mean_crossings(abp, ppg)
    is_larger = len(beats_a) + len(beats_p) > len(abp_beats) + len(beats_p)
    is_closer = abs(len(beats_a) - len(beats_p)) < abs(len(abp_beats) - len(ppg_beats))
    if is_larger and is_closer:
        # print(f"Mimic beats - {len(abp_beats), len(ppg_beats)},"
        #       f" Mean Crossing beats - {len(beats_a), len(beats_p)} ")
        abp_beats = beats_a
        ppg_beats = beats_p
        # plot_abp_ppg_with_pulse(seg_name + ' PEAKS (downward mean crossings)', abp, abp_beats, ppg, ppg_beats, fs)

    # Second iteration: Peak finding (SP manual methods)
    abp_beat_interval = len(abp) / len(abp_beats)
    ppg_beat_interval = len(ppg) / len(ppg_beats)
    abp_beats, _ = sp.find_peaks(-abp, distance=abp_beat_interval * .75, prominence=0.5)
    ppg_beats, _ = sp.find_peaks(-ppg, distance=ppg_beat_interval * .75, prominence=0.01)
    # plot_abp_ppg_with_pulse(seg_name + ' DIPS (sp.find_peaks)', abp, abp_beats, ppg, ppg_beats, fs)

    # Signal synchronization : delay approx = 18 (288 ms)
    abp, ppg = synchronization(abp, ppg, abp_beats, ppg_beats)
    # plot_abp_ppg(seg_name + ' Synchronised', abp, ppg, fs)

    # Third iteration: Peak finding
    abp_beats, _ = sp.find_peaks(abp, distance=abp_beat_interval * .75, prominence=0.5)
    ppg_beats, _ = sp.find_peaks(ppg, distance=ppg_beat_interval * .75, prominence=0.01)
    # plot_abp_ppg_with_pulse(seg_name + ' PEAKS (sp.find_peaks)', abp, abp_beats, ppg, ppg_beats, fs)
    normal_length_a, normal_length_p = len(abp_beats), len(ppg_beats)

    # Beat grouping
    abp_beats, ppg_beats = group_beats(abp_beats, ppg_beats)
    # plot_abp_ppg_with_pulse(seg_name + ' Grouped', abp, abp_beats, ppg, ppg_beats, fs)

    if max(normal_length_a, normal_length_p) > max(len(abp_beats), len(ppg_beats)) * 1.05:
        print(f"too big of a difference after grouping -"
              f"{max(normal_length_a, normal_length_p) - max(len(abp_beats), len(ppg_beats))}")

    abp_hr = len(abp_beats) / (len(abp) / fs) * 60
    ppg_hr = len(ppg_beats) / (len(ppg) / fs) * 60
    print(f"ABP beats - {len(abp_beats)}, Heart Rate - {abp_hr}")
    print(f"PPG beats - {len(ppg_beats)}, Heart Rate - {ppg_hr}")

    return {
        'abp': abp,
        'ppg': ppg,
        'abp_beats': abp_beats,
        'ppg_beats': ppg_beats,
        'abp_hr': abp_hr,
        'ppg_hr': ppg_hr
    }


def synchronization(abp, ppg, abp_dips, ppg_dips):
    # Find closest PPG dip to first ABP dip
    for i in range(0, len(abp_dips)):
        first_abp = abp_dips[i]
        differences = ppg_dips - first_abp
        smallest_diff = min(num for num in differences if num > 0)
        if smallest_diff < 30:
            first_ppg_ind, = np.argwhere(differences == smallest_diff)[0]
            break
    first_ppg = ppg_dips[first_ppg_ind]
    # print([first_abp_ind, first_abp], [first_ppg_ind, first_ppg])

    # Find closest ABP dip to last PPG dip
    last_ppg_ind = len(ppg_dips) - 1
    last_ppg = ppg_dips[last_ppg_ind]
    differences = abs(abp_dips - last_ppg)
    last_abp_ind = np.argmin(differences)
    last_abp = abp_dips[last_abp_ind]
    # print([last_abp_ind, last_abp], [last_ppg_ind, last_ppg])

    # Splice original ABP and PPG arrays
    new_abp = abp[first_abp:last_abp]
    new_ppg = ppg[first_ppg:last_ppg]
    # print(len(new_abp), len(new_ppg))

    return new_abp, new_ppg


def group_beats(abp_beats, ppg_beats):
    i = 0
    while i < min(len(abp_beats), len(ppg_beats)):
        a = abp_beats[i]
        p = ppg_beats[i]
        if p - 20 <= a <= p + 20:
            i += 1
        else:
            if a < p:
                # print(f"remove abp beat {a}")
                abp_beats = abp_beats[abp_beats != a]
                # if not value_exists(ppg_beats, a):
                #     ppg_beats = ppg_beats[ppg_beats != p]
            elif a > p:
                # print(f"remove ppg beat {p}")
                ppg_beats = ppg_beats[ppg_beats != p]
                # if not value_exists(abp_beats, p):
                #     abp_beats = abp_beats[abp_beats != a]
    abp_beats, ppg_beats = equal_out_by_shortening(abp_beats, ppg_beats)
    return abp_beats, ppg_beats


def value_exists(beats, a):
    v = True
    for x in beats:
        if x - 10 <= a <= x + 10:
            v = True
            break
        else:
            v = False
    return v


def get_beats_from_mean_crossings(abp, ppg):
    mean_value = np.mean(abp)
    above_mean = abp > mean_value
    count_a = np.count_nonzero(np.diff(above_mean.astype(int)) == -1)
    mean_value = np.mean(ppg)
    above_mean = ppg > mean_value
    count_p = np.count_nonzero(np.diff(above_mean.astype(int)) == -1)
    abp_beat_interval = len(abp) / ((count_a + count_p) / 2)
    ppg_beat_interval = abp_beat_interval

    abp_beats, _ = sp.find_peaks(-abp, distance=abp_beat_interval, prominence=5)
    ppg_beats, _ = sp.find_peaks(-ppg, distance=ppg_beat_interval, prominence=0.01)

    return abp_beats, ppg_beats


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
