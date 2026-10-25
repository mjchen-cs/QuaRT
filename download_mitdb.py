"""Download the MIT-BIH Arrhythmia Database record by record (resumable).

Usage:  python download_mitdb.py          # downloads only missing / truncated records
Re-run it until it reports 48/48 complete."""
import os, sys, time
import wfdb
from quart.ecg import MITDB_RECORDS, mitdb_missing

DIR = 'data/mitdb'


def main():
    os.makedirs(DIR, exist_ok=True)
    for attempt in range(1, 6):
        missing = mitdb_missing(DIR)
        if not missing:
            break
        print(f'[attempt {attempt}] {len(missing)} records to download: {missing}', flush=True)
        for r in missing:
            # remove truncated files so wfdb downloads them again
            for e in ('.hea', '.atr', '.dat'):
                p = os.path.join(DIR, f'{r}{e}')
                if os.path.exists(p) and (e != '.dat' or os.path.getsize(p) != 1950000):
                    if e == '.dat' or os.path.getsize(p) == 0:
                        os.remove(p)
            try:
                t = time.time()
                wfdb.dl_database('mitdb', DIR, records=[str(r)], annotators=['atr'], overwrite=False)
                print(f'  record {r} ok ({time.time()-t:.0f}s)', flush=True)
            except Exception as ex:
                print(f'  record {r} failed: {ex}', flush=True)
    missing = mitdb_missing(DIR)
    print(f'MIT-BIH: {len(MITDB_RECORDS) - len(missing)}/{len(MITDB_RECORDS)} records complete.')
    if missing:
        print('Still missing:', missing, '-> re-run this script.')
        sys.exit(1)


if __name__ == '__main__':
    main()
