import sys
import scipy as sp
from pathlib import Path
import wfdb
from matplotlib import pyplot as plt
import numpy as np
from IV_signal_visualization import load_data_from_segment
from V_signal_filtering import extract_ppg_data_from_segment, filter_ppg_sos_butterworth, filter_ppg_sos_chebyshev

# The name of the MIMIC-IV Waveform Database on PhysioNet
database_name = 'mimic4wdb/0.1.0'

# Segment for analysis
segment_names = ['83404654_0005', '82924339_0007', '84248019_0005', '82439920_0004', '82800131_0002', '84304393_0001',
                 '89464742_0001', '88958796_0004', '88995377_0001', '85230771_0004', '86643930_0004', '81250824_0005',
                 '87706224_0003', '83058614_0005', '82803505_0017', '88574629_0001', '87867111_0012', '84560969_0001',
                 '87562386_0001', '88685937_0001', '86120311_0001', '89866183_0014', '89068160_0002', '86380383_0001',
                 '85078610_0008', '87702634_0007', '84686667_0002', '84802706_0002', '81811182_0004', '84421559_0005',
                 '88221516_0007', '80057524_0005', '84209926_0018', '83959636_0010', '89989722_0016', '89225487_0007',
                 '84391267_0001', '80889556_0002', '85250558_0011', '84567505_0005', '85814172_0007', '88884866_0005',
                 '80497954_0012', '80666640_0014', '84939605_0004', '82141753_0018', '86874920_0014', '84505262_0010',
                 '86288257_0001', '89699401_0001', '88537698_0013', '83958172_0001']
segment_dirs = [database_name + '/waves/p100/p10020306/83404654', database_name + '/waves/p101/p10126957/82924339',
                database_name + '/waves/p102/p10209410/84248019', database_name + '/waves/p109/p10952189/82439920',
                database_name + '/waves/p111/p11109975/82800131', database_name + '/waves/p113/p11392990/84304393',
                database_name + '/waves/p121/p12168037/89464742', database_name + '/waves/p121/p12173569/88958796',
                database_name + '/waves/p121/p12188288/88995377', database_name + '/waves/p128/p12872596/85230771',
                database_name + '/waves/p129/p12933208/86643930', database_name + '/waves/p130/p13016481/81250824',
                database_name + '/waves/p132/p13240081/87706224', database_name + '/waves/p136/p13624686/83058614',
                database_name + '/waves/p137/p13791821/82803505', database_name + '/waves/p141/p14191565/88574629',
                database_name + '/waves/p142/p14285792/87867111', database_name + '/waves/p143/p14356077/84560969',
                database_name + '/waves/p143/p14363499/87562386', database_name + '/waves/p146/p14695840/88685937',
                database_name + '/waves/p149/p14931547/86120311', database_name + '/waves/p151/p15174162/89866183',
                database_name + '/waves/p153/p15312343/89068160', database_name + '/waves/p153/p15342703/86380383',
                database_name + '/waves/p155/p15552902/85078610', database_name + '/waves/p156/p15649186/87702634',
                database_name + '/waves/p158/p15857793/84686667', database_name + '/waves/p158/p15865327/84802706',
                database_name + '/waves/p158/p15896656/81811182', database_name + '/waves/p159/p15920699/84421559',
                database_name + '/waves/p160/p16034243/88221516', database_name + '/waves/p165/p16566444/80057524',
                database_name + '/waves/p166/p16644640/84209926', database_name + '/waves/p167/p16709726/83959636',
                database_name + '/waves/p167/p16715341/89989722', database_name + '/waves/p168/p16818396/89225487',
                database_name + '/waves/p170/p17032851/84391267', database_name + '/waves/p172/p17229504/80889556',
                database_name + '/waves/p173/p17301721/85250558', database_name + '/waves/p173/p17325001/84567505',
                database_name + '/waves/p174/p17490822/85814172', database_name + '/waves/p177/p17738824/88884866',
                database_name + '/waves/p177/p17744715/80497954', database_name + '/waves/p179/p17957832/80666640',
                database_name + '/waves/p180/p18080257/84939605', database_name + '/waves/p181/p18109577/82141753',
                database_name + '/waves/p183/p18324626/86874920', database_name + '/waves/p187/p18742074/84505262',
                database_name + '/waves/p188/p18824975/86288257', database_name + '/waves/p191/p19126489/89699401',
                database_name + '/waves/p193/p19313794/88537698', database_name + '/waves/p196/p19619764/83958172']


def main():
    # Segment 3 and 8 are helpful
    rel_segment_n = 3
    rel_segment_name = segment_names[rel_segment_n]
    rel_segment_dir = segment_dirs[rel_segment_n]

    # time since the start of the segment at which to begin extracting data
    start_seconds = 100
    no_seconds_to_load = 5

    segment_metadata = wfdb.rdheader(record_name=rel_segment_name,
                                     pn_dir=rel_segment_dir)
    print(f"Metadata loaded from segment: {rel_segment_name}")

    fs = round(segment_metadata.fs)
    sampfrom = fs * start_seconds
    sampto = fs * (start_seconds + no_seconds_to_load)

    segment_data = load_data_from_segment(fs, start_seconds, no_seconds_to_load, rel_segment_name, rel_segment_dir)

    print(f"{no_seconds_to_load} seconds of data extracted from: {rel_segment_name}")

    ppg, fs = extract_ppg_data_from_segment(segment_data)

    # filter cut-offs
    lpf_cutoff = 0.7  # Hz
    hpf_cutoff = 10  # Hz

    # create filter
    sos_ppg, w, h = filter_ppg_sos_butterworth(lpf_cutoff, hpf_cutoff, segment_data, fs)

    # filter PPG
    ppg_filt = sp.signal.sosfiltfilt(sos_ppg, ppg)

    # Plot original and filtered PPG
    # fig, ax = plt.subplots()
    # t = np.arange(0, len(ppg_filt)) / segment_data.fs
    #
    # ax.plot(t, ppg,
    #         linewidth=2.0,
    #         label="original PPG")
    #
    # ax.plot(t, ppg_filt,
    #         linewidth=2.0,
    #         label="filtered PPG")
    #
    # ax.set(xlim=(0, no_seconds_to_load))
    # plt.xlabel('time (s)')
    # plt.ylabel('PPG')
    #
    # plt.legend()
    # plt.show()

    # Calculate first derivative
    d1ppg = sp.signal.savgol_filter(ppg_filt, 9, 5, deriv=1)

    # Calculate second derivative
    d2ppg = sp.signal.savgol_filter(ppg_filt, 9, 5, deriv=2)

    t = np.arange(0, len(ppg_filt)) / segment_data.fs

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False)
    ax1.plot(t, ppg_filt)
    ax1.set(xlabel='', ylabel='PPG')

    plt.suptitle('The PPG signal and its first and second derivatives')

    ax2.plot(t, d1ppg)
    ax2.set(xlabel='',
            ylabel='PPG\'')

    ax3.plot(t, d2ppg)
    ax3.set(xlabel='Time (s)',
            ylabel='PPG\'\'')

    plt.show()


if __name__ == "__main__":
    main()
