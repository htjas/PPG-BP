import wfdb
from matplotlib import pyplot as plt
import numpy as np


def plot_wfdb_segment(segment_name, segment_data):
    title_text = f"Segment {segment_name}"
    wfdb.plot_wfdb(record=segment_data,
                   title=title_text,
                   time_units='seconds')


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
