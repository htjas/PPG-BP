import pandas as pd
from pprint import pprint
from I_exploration import load_records
import wfdb

database_name = 'mimic4wdb/0.1.0'
records = load_records(database_name)

print(f"Earlier, we loaded {len(records)} records from the '{database_name}' database.")

required_sigs = ['ABP', 'Pleth']

# convert from minutes to seconds
req_seg_duration = 10 * 60

matching_recs = {'dir': [], 'seg_name': [], 'length': []}

for record in records:
    print('Record: {}'.format(record), end="", flush=True)
    record_parent = str(record.parent).replace('\\', '/')
    record_dir = f'{database_name}/{record_parent}'
    record_name = record.name
    print(' (reading data)')
    record_data = wfdb.rdheader(record_name,
                                pn_dir=record_dir,
                                rd_segments=True)

    # Check whether the required signals are present in the record
    sigs_present = record_data.sig_name
    if not all(x in sigs_present for x in required_sigs):
        print('   (missing signals)')
        continue

    # Get the segments for the record
    segments = record_data.seg_name

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

        # Next check that all required signals are present in the segment
        sigs_present = segment_metadata.sig_name

        if all(x in sigs_present for x in required_sigs):
            matching_recs['dir'].append(record_dir)
            matching_recs['seg_name'].append(segment)
            matching_recs['length'].append(seg_length)
            print(' (met requirements)')
            # Since we only need one segment per record break out of loop
            break
        else:
            print(' (long enough, but missing signal(s))')

print(f"A total of {len(matching_recs['dir'])} records met the requirements:")

relevant_segments_names = "\n - ".join(matching_recs['seg_name'])
print(f"\nThe relevant segment names are:\n - {relevant_segments_names}")

relevant_dirs = "\n - ".join(matching_recs['dir'])
print(f"\nThe corresponding directories are: \n - {relevant_dirs}")

df_matching_recs = pd.DataFrame(data=matching_recs)
df_matching_recs.to_csv('matching_records.csv', index=False)
p = 1
