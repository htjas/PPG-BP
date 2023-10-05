import wfdb
from pathlib import Path
import pandas as pd


def load_records(db_name):
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
    df_matching_recs = pd.DataFrame(data=records)
    df_matching_recs.to_csv('matching_records.csv', index=False)
