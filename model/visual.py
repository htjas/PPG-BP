import wfdb
from matplotlib import pyplot as plt
import numpy as np


def plot_wfdb_segment(segment_name, segment_data):
    title_text = f"Segment {segment_name}"
    wfdb.plot_wfdb(record=segment_data,
                   title=title_text,
                   time_units='seconds')


def plot_features(seg_name, x1, y1, x2, y2, x3, y3, x4, y4):
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   sharex=False,
                                   sharey=False,
                                   figsize=(8, 8))
    fig.suptitle(f"Segment {seg_name}")

    ax1.scatter(np.arange(len(y1)), y1, color='red', label='Sys')
    ax1.scatter(np.arange(len(y2)), y2, color='blue', label='Dia')
    ax1.set_title("ABP")
    # ax1.set_xlim([0, 100])

    ax2.scatter(np.arange(len(y3)), y3, color='red', label='Sys')
    ax2.scatter(np.arange(len(y4)), y4, color='blue', label='Dia')
    ax2.set_title("PPG")
    # ax2.set_xlim([0, 100])

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
    fig.suptitle(f"{segment_name}")

    t = np.arange(0, (len(abp) / fs), 1.0 / fs)
    if len(t) != len(abp):
        diff = len(t) - len(abp)
        t = t[0:-diff]
    ax1.plot(t, abp, color='red', label='ABP')
    ax1.set_title("ABP")
    ax1.set_xlim([-0.1, 10])
    # ax1.set_ylim([-40, 140])

    t = np.arange(0, (len(ppg) / fs), 1.0 / fs)
    if len(t) != len(ppg):
        diff = len(t) - len(ppg)
        t = t[0:-diff]
    ax2.plot(t, ppg, color='green', label='PPG')
    ax2.set_title("PPG")
    ax2.set_xlim([-0.1, 10])
    # ax2.set_ylim([-0.3, 1.1])

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
    fig.suptitle(f"{segment_name}")

    t = np.arange(0, (len(abp) / fs), 1.0 / fs)
    if len(t) != len(abp):
        diff = len(t) - len(abp)
        t = t[0:-diff]

    ax1.plot(t, abp, color='red')
    ax1.scatter(t[0] + abp_beats / fs,
                abp[abp_beats],
                color='black',
                marker='o')
    # Dotted lines for synchronisation purposes
    # y_limits = ax1.get_ylim()
    # for beat in abp_beats:
    #     ax1.plot([t[beat], t[beat]], [y_limits[0], abp[beat]], color='red', linestyle='--')
    ax1.set_xlim([-0.1, 10])
    ax1.set_title('ABP with IBIS')

    t = np.arange(0, (len(ppg) / fs), 1.0 / fs)
    if len(t) != len(ppg):
        diff = len(t) - len(ppg)
        t = t[0:-diff]
    ax2.plot(t, ppg, color='green')
    ax2.scatter(t[0] + ppg_beats / fs,
                ppg[ppg_beats],
                color='black',
                marker='o')
    # Dotted lines for synchronisation purposes
    # y_limits = ax2.get_ylim()
    # for beat in ppg_beats:
    #     ax2.plot([t[beat], t[beat]], [y_limits[1], ppg[beat]], color='green', linestyle='--')
    ax2.set_xlim([-0.1, 10])
    ax2.set_title('PPG with IBIS')

    plt.show()
