import os
import pathlib
import pandas as pd
from visual import *
from init_scripts import *


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

        # print(f"File {i} / {round(len(filenames)/2)} - {filename}")
        df = pd.read_csv(f"{folder}/abp_{seg_name}.{end}")
        abp = df.values
        # print(f"abp signal of segment - {seg_name}, tenth value - {abp[10]}")

        df = pd.read_csv(f"{folder}/ppg_{seg_name}.{end}")
        ppg = df.values
        # print(f"ppg signal of segment - {seg_name}, tenth value - {ppg[10]}")

        i += 1


def split_filename(filename):
    x = filename.split("_", 1)
    y = x[1].split('.')
    sig = x[0]
    seg_name = y[0]
    end = y[1]
    return sig, seg_name, end


def main():
    process_data('data', 62.4725)


if __name__ == "__main__":
    main()
