import wfdb
from pathlib import Path
import pandas as pd


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
        record_dir = f'{database_name}/{record.parent}'
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
                print(' (met requirements)')
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


def get_abp_pleth_col_no(segment_data):
    """
    Method to find out which column in the record PPG and ABP are recorded in
    :param segment_data: the segment data of a record
    :return: abp and pleth columns in integer
    """
    abp_col = 0
    pleth_col = 0
    for sig_no in range(0, len(segment_data.sig_name)):
        if "ABP" in segment_data.sig_name[sig_no]:
            abp_col = sig_no
        if "Pleth" in segment_data.sig_name[sig_no]:
            pleth_col = sig_no
    return abp_col, pleth_col
