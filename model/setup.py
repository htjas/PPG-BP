import wfdb
from pathlib import Path


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
        if su == 5:
            break
        for study in studies:
            loaded_records.append(Path(f'{subject}{study}'))
            st = st + 1
            print(f"Study {st}/{len(studies)} - {study}")

    print(f"Loaded {len(loaded_records)} records from the '{db_name}' database")
    return loaded_records
