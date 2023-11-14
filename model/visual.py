import wfdb
from matplotlib import pyplot as plt
import numpy as np


def plot_wfdb_segment(segment_name, segment_data):
    title_text = f"Segment {segment_name}"
    wfdb.plot_wfdb(record=segment_data,
                   title=title_text,
                   time_units='seconds')


def plot_trio(seg_name, x1, y1, x2, y2, x3, y3):
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                                 sharex=False,
                                                 sharey=False,
                                                 figsize=(8, 8))
    fig.suptitle(f"Segment {seg_name}")

    ax1.scatter(x1, y1, color='black', label='Agi')
    ax1.set_title("Agi")
    ax1.set_xlim([0, 1000])

    ax2.scatter(x2, y2, color='red', label='Sys')
    ax2.scatter(x3, y3, color='blue', label='Dia')
    ax2.set_title("Sys + Dia")
    ax2.set_xlim([0, 1000])

    plt.show()


def plot_quad(seg_name, sig, fs, raw, filt, d1, d2):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                                 sharex=False,
                                                 sharey=False,
                                                 figsize=(8, 8))
    fig.suptitle(f"Segment {seg_name} - {sig}")

    t = np.arange(0, (len(raw) / fs), 1.0 / fs)

    ax1.plot(t, raw, color='black', label='Raw')
    ax1.set_title("Raw")
    ax1.set_xlim([55, 60])

    ax2.plot(t, filt, color='black', label='Filt')
    ax2.set_title("Filtered")
    ax2.set_xlim([55, 60])

    ax3.plot(t, d1, color='black', label='Filt D1')
    ax3.set_title("Filtered D1")
    ax3.set_xlim([55, 60])

    ax4.plot(t, d2, color='black', label='Filt D2')
    ax4.set_title("Filtered D2")
    ax4.set_xlim([55, 60])

    plt.show()


def plot_extracted_data(seg_name, x1, y1, x2, y2, x3, y3):
    plt.suptitle(seg_name)
    plt.scatter(x1, y1, color='black')
    plt.scatter(x2, y2, color='red')
    plt.scatter(x3, y3, color='blue')
    plt.xlim([0, 1000])
    plt.show()


def plot_abp_ppg(segment_name, abp, ppg, fs):
    """
    Plot the simultaneous PPG and ABP graphs
    :param segment_name: name of the segment
    :param ppg: PPG values array
    :param abp: ABP values array
    :param fs: Sampling frequency
    """
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   sharex=False,
                                   sharey=False,
                                   figsize=(8, 8))
    fig.suptitle(f"Segment {segment_name}")

    t = np.arange(0, (len(ppg) / fs), 1.0 / fs)

    ax1.plot(t, abp, color='red', label='ABP')
    ax1.set_title("ABP")
    ax1.set_xlim([0, 60])

    ax2.plot(t, ppg, color='green', label='PPG')
    ax2.set_title("PPG")
    ax2.set_xlim([0, 60])

    plt.show()

    # t = np.arange(0, (len(ppg) / fs), 1.0 / fs)
    #
    # plt.plot(t, ppg, color='black', label='PPG')
    # plt.xlim([50, 55])
    # plt.show()


def plot_abp_ppg_with_pulse(segment_name, abp, abp_beats, ppg, ppg_beats, fs):
    """
    Plot the simultaneous PPG and ABP graphs
    :param ppg_beats: points of detected beats along the PPG graph
    :param abp_beats: points of detected beats along the ABP graph
    :param segment_name: name of the segment
    :param ppg: PPG values array
    :param abp: ABP values array
    :param fs: Sampling frequency
    """
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   sharex=False,
                                   sharey=False,
                                   figsize=(8, 8))
    fig.suptitle(f"Segment {segment_name}")

    t = np.arange(0, (len(ppg) / fs), 1.0 / fs)

    ax1.plot(t, abp, color='red')
    ax1.scatter(t[0] + abp_beats / fs,
                abp[abp_beats],
                color='black',
                marker='o')
    ax1.set_xlim([0, 5])
    ax1.set_title('ABP with IBIS')

    ax2.plot(t, ppg, color='green')
    ax2.scatter(t[0] + ppg_beats / fs,
                ppg[ppg_beats],
                color='black',
                marker='o')
    ax2.set_xlim([0, 5])
    ax2.set_title('PPG with IBIS')

    plt.show()
