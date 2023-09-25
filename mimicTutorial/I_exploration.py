import sys
from pathlib import Path
import wfdb


def load_records(db_name):
    subjects = wfdb.get_record_list(db_name)
    print(f"The '{db_name}' database contains data from {len(subjects)} subjects")

    max_records_load = 200
    loaded_records = []
    for subject in subjects:
        studies = wfdb.get_record_list(f'{db_name}/{subject}')
        for study in studies:
            loaded_records.append(Path(f'{subject}{study}'))
            if len(loaded_records) >= max_records_load:
                print("Reached maximum records")
                break

    print(f"Loaded {len(loaded_records)} records from the '{db_name}' database")
    return loaded_records


def main():
    database_name = 'mimic4wdb/0.1.0'
    records = load_records(database_name)

    last_five_records = [str(x) for x in records[-5:]]
    last_five_records = "\n - ".join(last_five_records)
    print(f"Last five records: \n - {last_five_records}")

    print("""
    Note the formatting of these records:
     - intermediate directory ('p100' in this case)
     - subject identifier (e.g. 'p10014354')
     - record identifier (e.g. '81739927'
     """)

    idx = 3
    record = records[idx]
    record_parent = str(record.parent).replace('\\', '/')
    record_dir = f"{database_name}/{record_parent}"
    print("PhysioNet directory specified for record: {}".format(record_dir))

    record_name = record.name
    print("Record name: {}".format(record_name))

    record_data = wfdb.rdheader(record_name, pn_dir=record_dir, rd_segments=True)
    remote_url = "https://physionet.org/content/" + record_dir + "/" + record_name + ".hea"
    print(f"Done: metadata loaded for record '{record_name}' from the header file at:\n{remote_url}")

    print("- Number of signals: {}".format(record_data.n_sig))
    print(f"- Duration: {record_data.sig_len / (record_data.fs * 60 * 60):.1f} hours")
    print(f"- Base sampling frequency: {record_data.fs} Hz")

    segments = record_data.seg_name
    print(f"The {len(segments)} segments from record {record_name} are:\n{segments}")

    segment_metadata = wfdb.rdheader(record_name=segments[5], pn_dir=record_dir)

    print(f"""Header metadata loaded for: 
    - the segment '{segments[5]}'
    - in record '{record_name}'
    - for subject '{str(Path(record_dir).parent.parts[-1])}'
    """)

    print(f"This segment contains the following signals: {segment_metadata.sig_name}")
    print(f"The signals are measured in units of: {segment_metadata.units}")

    print(f"The signals have a base sampling frequency of {segment_metadata.fs:.1f} Hz")
    print(f"and they last for {segment_metadata.sig_len / (segment_metadata.fs * 60):.1f} minutes")


if __name__ == "__main__":
    main()
