"""Recompute partial_correlation for all existing ROI files using new Ledoit-Wolf impl."""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from datasets.compute_roi_fc import compute_partial_correlation

import warnings
warnings.filterwarnings('ignore')


def _worker(args):
    roi_path, out_path = args
    try:
        roi = np.load(roi_path)
        pc = compute_partial_correlation(roi)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, pc)
        return (roi_path, 'ok')
    except Exception as e:
        return (roi_path, f'error: {e}')


def find_jobs(roots):
    jobs = []
    for root in roots:
        roi_root = os.path.join(root, 'roi')
        if not os.path.isdir(roi_root):
            continue
        for atlas in os.listdir(roi_root):
            adir = os.path.join(roi_root, atlas)
            if not os.path.isdir(adir):
                continue
            for fname in os.listdir(adir):
                if not fname.endswith('.npy'):
                    continue
                # ROI files: <subj>_<atlas>.npy  -> partial: <subj>_<atlas>_partial_correlation.npy
                stem = fname[:-4]  # strip .npy
                roi_path = os.path.join(adir, fname)
                out_path = os.path.join(root, 'fc', atlas, 'partial_correlation',
                                        f'{stem}_partial_correlation.npy')
                if os.path.isfile(out_path):
                    continue
                jobs.append((roi_path, out_path))
    return jobs


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--roots', nargs='+', default=None,
                   help='dataset root(s) containing roi/ and fc/. '
                        'If omitted, defaults to all known local datasets.')
    p.add_argument('--workers', type=int, default=int(os.environ.get('N_WORKERS', '8')))
    args = p.parse_args()

    roots = args.roots or [
        '/data_ssd/cwang/fmri/adhd200_preprocessed',
        '/data_ssd/cwang/fmri/cobre_preprocessed',
        '/data_ssd/cwang/fmri/hcpep_preprocessed',
        '/data_ssd/cwang/fmri/hcpya_preprocessed',
        '/data_ssd/cwang/fmri/ucla_preprocessed',
        '/data_ssd/cwang/fmri/hcpa_processed',
        '/data_ssd/cwang/fmri/hcpd_preprocessed',
        '/data_ssd/cwang/fmri/abcd_preprocessed',
        '/data_ssd/cwang/fmri/ukb_preprocessed',
        '/data_ssd/cwang/fmri/abide_preprocessed',
    ]

    jobs = find_jobs(roots)
    print(f'jobs: {len(jobs)}', flush=True)
    if not jobs:
        sys.exit(0)

    n_workers = args.workers
    t0 = time.time()
    ok = err = 0
    with Pool(processes=n_workers) as pool:
        for i, (path, status) in enumerate(pool.imap_unordered(_worker, jobs, chunksize=8), 1):
            if status == 'ok':
                ok += 1
            else:
                err += 1
                print(f'[err] {path}: {status}', flush=True)
            if i % 500 == 0 or i == len(jobs):
                rate = i / max(time.time() - t0, 1e-6)
                eta = (len(jobs) - i) / max(rate, 1e-6)
                print(f'[progress] {i}/{len(jobs)} ok={ok} err={err} '
                      f'rate={rate:.0f}/s eta={eta/60:.1f}min', flush=True)
    print(f'done: ok={ok} err={err} in {(time.time()-t0)/60:.1f}min')
