import time

import wfdb
from pathlib import Path
import pandas as pd
from visual import *
import os
import logging
import random


def load_filter_and_save_records(db_name, path, records_to_load, single_record_arrays):
    """
    Load records from database,
     filter them directly according to criteria found in filter_single_record() method,
     save them to the given path
    """
    current_records = os.listdir(f'.{path}abp')
    record_names = [item.split('_')[1] for item in current_records]

    subjects = wfdb.get_record_list(db_name)
    random.shuffle(subjects)
    print(f"The '{db_name}' database contains data from {len(subjects)} subjects ({time.strftime('%H:%M:%S')})")
    path = os.path.abspath(os.getcwd()) + path
    records = []
    su = 0
    for subject in subjects:
        # if subject != '36/3618396/':
        #     su += 1
        #     continue
        subject_name = subject.split('/')[1]
        if subject_name in record_names:
            print(f"    (already in saved records, skipping)")
            continue
        su = su + 1
        print(f"Subject {su}/{len(subjects)} - {subject}")
        studies = wfdb.get_record_list(f'{db_name}/{subject}')
        # for study in studies:
        study = studies[0]
        p = Path(f"{subject}/{study}")
        res = filter_and_save_single_subject(p, db_name, path, single_record_arrays)
        if res is not None:
            records = os.listdir(path + 'abp')
            print(f"-------------------------------- so far {len(records)} records saved")
            if len(records) > records_to_load:
                break

    print(f"Loaded {len(records)} records from the '{db_name}' database")


def filter_and_save_single_subject(subject, database_name, path, no_records):
    """
    Filter records of a subject according to criteria:
        - ABP and Pleth signals present
        - 10 min of continuous signal available
        - Signals do not contain faulty values (if np.isnan(value) or np.isinf(value) or value == 0.0)
        - ABP values are between 30 and 250
        - PPG values are between 0 and 1
    """
    if database_name == 'mimic4wdb/0.1.0':
        required_sigs = ['ABP', 'Pleth']  # 'Pleth' for MIMIC4 'PLETH' for MIMIC3
    else:
        required_sigs = ['ABP', 'PLETH']
    req_seg_duration = 10 * 60 + 5

    try:
        # print(f"Record {record.name}")
        record_dir = f'{database_name}/{subject.parent}'.replace("\\", "/")
        record_data = wfdb.rdheader(subject.name, pn_dir=record_dir, rd_segments=True)

        signal_names = record_data.sig_name
        if not all(x in signal_names for x in required_sigs):
            print('   (missing signals)')
            return None

        if hasattr(record_data, 'seg_name'):
            tot_arr_count = 0
            segments = record_data.seg_name
            gen = (segment for segment in segments if segment != '~')
            for segment in gen:
                if tot_arr_count == no_records:
                    print(f" - reached {no_records} records limit from single subject")
                    break
                print(' - Segment: {}'.format(segment), end="", flush=True)
                segment_metadata = wfdb.rdheader(record_name=segment,
                                                 pn_dir=record_dir)
                tot_seg_length = segment_metadata.sig_len
                fs = segment_metadata.fs
                seg_duration = tot_seg_length / fs

                if seg_duration < req_seg_duration:
                    print(f' (too short at {seg_duration / 60:.1f} mins)')
                    continue

                sigs_present = segment_metadata.sig_name
                if all(x in sigs_present for x in required_sigs):
                    print(f" (met requirements)  Segment length - {seg_duration / 60:.1f}"
                          f", checking for consecutive non-faulty values ({time.strftime('%H:%M:%S')})",
                          end="", flush=True)
                else:
                    print(' (long enough, but missing signal(s))')
                    continue

                segment_data = wfdb.rdrecord(record_name=segment,
                                             sampfrom=0,
                                             sampto=tot_seg_length,
                                             pn_dir=record_dir)
                abp_col, pleth_col = get_abp_pleth_col_no(segment_data, required_sigs[1])
                abp = segment_data.p_signal[:, abp_col]
                ppg = segment_data.p_signal[:, pleth_col]

                req_cons = int(fs * req_seg_duration)
                current_count, cons_count, max_cons_count = 0, 0, 0
                abp_cons, ppg_cons = np.array([]), np.array([])

                for i, (a, p) in enumerate(zip(abp, ppg)):
                    if tot_arr_count == no_records:
                        break
                    if (
                            30 < a < 250 and
                            0 < p < 1 and
                            not (np.isnan(a) or np.isinf(a) or a == 0.0) and
                            not (np.isnan(p) or np.isinf(p) or p == 0.0)
                    ):
                        current_count += 1
                        if current_count > max_cons_count:
                            max_cons_count = current_count
                        if current_count >= req_cons:
                            abp_cons = abp[i - req_cons + 1:i + 1]
                            ppg_cons = ppg[i - req_cons + 1:i + 1]

                            df_abp = pd.DataFrame(data=abp_cons)
                            df_abp.to_csv(f"{path}abp/abp_{segment}_{cons_count}.csv", index=False)

                            df_ppg = pd.DataFrame(data=ppg_cons)
                            df_ppg.to_csv(f"{path}ppg/ppg_{segment}_{cons_count}.csv", index=False)
                            current_count = 0
                            cons_count += 1
                            tot_arr_count += 1
                    else:
                        current_count = 0

                if len(abp_cons) == 0 or len(ppg_cons) == 0:
                    m = int(max_cons_count / fs)
                    print(f" - ✕ (max {m} seconds array found)")
                else:
                    print(f" - ✔ ({cons_count} array(s) of 10 min found and saved)")

    except Exception as e:
        print(f'    {e}')
        return None

    return True


def load_records(db_name):
    """
    Load all records from database
    """
    subjects = wfdb.get_record_list(db_name)
    print(f"The '{db_name}' database contains data from {len(subjects)} subjects")

    max_records_load = 200
    loaded_records = []
    su = 0
    for subject in subjects:
        studies = wfdb.get_record_list(f'{db_name}/{subject}')
        su = su + 1
        print(f"Subject {su}/{len(subjects)} - {subject}")
        st = 0
        # if su == 5:
        #     break
        for study in studies:
            loaded_records.append(Path(f'{subject}{study}'))
            st = st + 1
            print(f"Study {st}/{len(studies)} - {study}")

    print(f"Loaded {len(loaded_records)} records from the '{db_name}' database")
    return loaded_records


def filter_records(records, database_name):
    """
    Filter records according to criteria:
        - ABP and Pleth signals present
        - 10 min of continuous signal available
    """
    required_sigs = ['ABP', 'Pleth']
    matching_recs = {'seg_name': [], 'length': [], 'dir': []}
    rec = 0
    for record in records:
        print(f"Record {rec}/{len(records)} - {record.name}")
        rec = rec + 1
        record_dir = f'{database_name}/{record.parent}'  # .replace("\\", "/")
        record_data = wfdb.rdheader(record.name, pn_dir=record_dir, rd_segments=True)

        signal_names = record_data.sig_name

        if not all(x in signal_names for x in required_sigs):
            print('   (missing signals)')
            continue
        print(record_data.sig_name)

        segments = record_data.seg_name

        # convert from minutes to seconds
        req_seg_duration = 10 * 60

        # Check to see if the segment is 10 min long
        # If not, move to the next one
        gen = (segment for segment in segments if segment != '~')
        for segment in gen:
            print(' - Segment: {}'.format(segment), end="", flush=True)
            segment_metadata = wfdb.rdheader(record_name=segment,
                                             pn_dir=record_dir)
            seg_length = segment_metadata.sig_len / segment_metadata.fs

            if seg_length < req_seg_duration:
                print(f' (too short at {seg_length / 60:.1f} mins)')
                continue

            sigs_present = segment_metadata.sig_name

            if all(x in sigs_present for x in required_sigs):
                matching_recs['seg_name'].append(segment)
                matching_recs['length'].append(seg_length)
                matching_recs['dir'].append(record_dir)
                print(f" (met requirements)  Segment length - {seg_length / 60:.1f}")
                # # Since we only need one segment per record break out of loop
                # break
            else:
                print(' (long enough, but missing signal(s))')

    print(f"A total of {len(matching_recs['dir'])} records met the requirements:")

    relevant_segments_names = "\n - ".join(matching_recs['seg_name'])
    print(f"\nThe relevant segment names are:\n - {relevant_segments_names}")

    relevant_dirs = "\n - ".join(matching_recs['dir'])
    print(f"\nThe corresponding directories are: \n - {relevant_dirs}")

    return matching_recs


def save_records_to_csv(records):
    """
    Save filtered records to .csv File
    """
    df_matching_recs = pd.DataFrame(data=records)
    df_matching_recs.to_csv('matching_records.csv', index=False)


def load_filter_save_records(database_name):
    """
    Combine 3 methods of MIMIC data initialization
    :param database_name: name of the database to be loaded
    :return: Result is a .csv file of segments that match the criteria, saved in the parent directory
    """
    records = load_records(database_name)
    matching_records = filter_records(records, database_name)
    save_records_to_csv(matching_records)


def load_basic_data_from_segment(segment_name, segment_dir):
    """
    Load basic segment data
    """
    print("Loading basic segment data")
    segment_data = wfdb.rdrecord(record_name=segment_name, pn_dir=segment_dir)

    print(
        f"This segment contains waveform data for the following {segment_data.n_sig} signals: {segment_data.sig_name}")
    print(f"The signals are sampled at a base rate of {segment_data.fs} Hz (and some are sampled at multiples of this)")
    print(f"They last for {segment_data.sig_len / (60 * segment_data.fs):.1f} minutes")


def load_metadata_from_segment(rel_segment_name, rel_segment_dir):
    """
    Load and return metadata
    """
    segment_metadata = wfdb.rdheader(record_name=rel_segment_name, pn_dir=rel_segment_dir)
    # print(f'Metadata loaded from segment: {rel_segment_name}')
    return segment_metadata


def load_data_from_segment(fs, start_seconds, n_seconds_to_load, rel_segment_name, rel_segment_dir):
    """
    Load full data from
    :param fs: Sampling Frequency
    :param start_seconds: Start time of segment reading
    :param n_seconds_to_load: Segment length to load
    :param rel_segment_name: Segment name
    :param rel_segment_dir: Segment directory
    :return: Fully loaded segment data
    """
    samp_from = fs * start_seconds
    samp_to = fs * (start_seconds + n_seconds_to_load)

    segment_data = wfdb.rdrecord(record_name=rel_segment_name,
                                 sampfrom=samp_from,
                                 sampto=samp_to,
                                 pn_dir=rel_segment_dir)

    print(f"Data loaded from segment: {rel_segment_name}")
    return segment_data


def get_abp_pleth_col_no(segment_data, pleth):
    """
    Method to find out which column in the record PPG and ABP are recorded in
    :param segment_data: the segment data of a record
    :param pleth: name of the PPG column (different for MIMIC3 and MIMIC4)
    :return: abp and pleth columns in integer
    """
    abp_col = None
    pleth_col = None
    for sig_no in range(0, len(segment_data.sig_name)):
        if "ABP" in segment_data.sig_name[sig_no]:
            abp_col = sig_no
        if pleth in segment_data.sig_name[sig_no]:
            pleth_col = sig_no
    return abp_col, pleth_col


def extract_save_bp_ppg_data(segments, path):
    # Extract 10 minutes of simultaneous BP and PPG signals from each record
    i = 1
    for segment in segments:
        print(f"Segment {i} / {len(segments)}")

        segment_metadata = load_metadata_from_segment(segment[0], segment[2])

        segment_data = load_data_from_segment(round(segment_metadata.fs),
                                              0,
                                              # 600 doesn't work, incrementing issue
                                              605,
                                              segment[0],
                                              segment[2])

        # plot_wfdb_segment(segment[0], segment_data)

        abp_col, pleth_col = get_abp_pleth_col_no(segment_data, 'pleth')

        abp = segment_data.p_signal[:, abp_col]
        ppg = segment_data.p_signal[:, pleth_col]

        if check_if_faulty(abp) or check_if_faulty(ppg):
            i += 1
            continue

        fs = segment_data.fs

        path = os.path.abspath(os.getcwd()) + path

        df_abp = pd.DataFrame(data=abp)
        df_abp.to_csv(f"{path}abp_{segment[0]}.csv", index=False)

        df_ppg = pd.DataFrame(data=ppg)
        df_ppg.to_csv(f"{path}ppg_{segment[0]}.csv", index=False)

        # plot_abp_ppg(segment[0], abp, ppg, fs)
        i += 1


def check_if_faulty(data):
    for value in data:
        if value < 0 or value > 250:
            print(value)
            print("Segment contains faulty values")
            return True


def init_logger(filename):
    # Configure the logging settings
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Create a file handler and set the log level
    file_handler = logging.FileHandler(f'{filename}.log')
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    # Add the file handler to the root logger
    logging.getLogger('').addHandler(file_handler)

    return logging


def main():
    load_filter_and_save_records('mimic4wdb/0.1.0', '/mimic4/', 300, 10)

    # records = load_records('mimic4wdb/0.1.0')
    # matching_records = filter_records(records, 'mimic4wdb/0.1.0')
    # print(len(matching_records))


if __name__ == "__main__":
    main()
