import os
import pathlib
import pandas as pd
from visual import *
from init_scripts import *


def load_segment_data_from_csv():
    filenames = os.listdir('data')

    i = 1
    for filename in filenames:
        print(f"File {i} / {len(filenames)} - {filename}")

        df = pd.read_csv(f"data/{filename}")
        segment_values = df.values

        i += 1
        break


def main():
    load_segment_data_from_csv()


if __name__ == "__main__":
    main()
