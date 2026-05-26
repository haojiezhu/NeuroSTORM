#!/usr/bin/env python3
"""End-to-end ABIDE I + ABIDE II pipeline.

For each subject, download the preprocessed BOLD (+ mask) NIfTI from the
fcp-indi S3 bucket, run NeuroSTORM voxel preprocessing (2mm resample,
TR 0.8s resample, 96^3 center crop, z-norm, int8 quantization), then
extract ROI time series + FC matrices for every atlas in datasets/atlas/,
and finally delete the downloaded NIfTIs to bound disk use.

Sources:
  ABIDE I  -> data/Projects/ABIDE_Initiative/Outputs/cpac/filt_noglobal/
              {func_preproc,func_mask}/<SITE>_<SUBID>_*.nii.gz
  ABIDE II -> data/Projects/ABIDE2/Outputs/fmriprep/fmriprep/sub-XXXX/
              [ses-X/]func/*_space-MNI152NLin2009cAsym_desc-{preproc_bold,brain_mask}.nii.gz
  Phenotypic:
    ABIDE I  -> PhenotypicData/phenotypic_<SITE>.csv
    ABIDE II -> RawData/ABIDEII-<SITE>/participants.tsv

Output layout (matches other NeuroSTORM datasets):
  <save_root>/img/<subj>/data.pt
  <save_root>/roi/<atlas>/<subj>_<atlas>.npy
  <save_root>/fc/<atlas>/{correlation,partial_correlation}/<subj>_<atlas>_<fc>.npy
  <save_root>/metadata/phenotypic.csv

Usage:
  python scripts/dataset_download/preprocess_abide.py \
      --save_root /data_ssd/cwang/fmri/abide_preprocessed \
      --raw_dir   /data_ssd/cwang/fmri/abide_raw_tmp \
      --num_processes 6
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from datasets.compute_roi_fc import (  # noqa: E402
    compute_fc,
    extract_roi_from_volume,
    load_atlas,
    resize_atlas_labels,
)
from datasets.preprocessing_volume import (  # noqa: E402
    select_middle_96,
    spatial_resampling,
    temporal_resampling,
)

S3 = 'https://s3.amazonaws.com/fcp-indi'
NS = '{http://s3.amazonaws.com/doc/2006-03-01/}'

ABIDE1_PREFIX = 'data/Projects/ABIDE_Initiative/Outputs/cpac/filt_noglobal/'
ABIDE1_PHENO_PREFIX = 'data/Projects/ABIDE_Initiative/PhenotypicData/'
ABIDE2_FMRIPREP_PREFIX = 'data/Projects/ABIDE2/Outputs/fmriprep/fmriprep/'
ABIDE2_RAW_PREFIX = 'data/Projects/ABIDE2/RawData/'

DEFAULT_FC_TYPES = ('correlation', 'partial_correlation')


# ---------------------------------------------------------------------------
# S3 listing helpers (anonymous, public bucket)
# ---------------------------------------------------------------------------

def _http_get(url: str, retries: int = 5, timeout: int = 60) -> bytes:
    last_err: Exception | None = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            time.sleep(2 ** i)
    raise RuntimeError(f'GET failed after {retries}: {url} ({last_err})')


def list_bucket(prefix: str, delimiter: str | None = None) -> tuple[list[str], list[str]]:
    """Return (keys, common_prefixes) under the given S3 prefix."""
    keys: list[str] = []
    common: list[str] = []
    marker = ''
    while True:
        url = f'{S3}?prefix={prefix}&marker={marker}'
        if delimiter:
            url += f'&delimiter={delimiter}'
        x = _http_get(url)
        root = ET.fromstring(x)
        for c in root.findall(f'{NS}Contents'):
            k = c.find(f'{NS}Key').text
            keys.append(k)
        for p in root.findall(f'{NS}CommonPrefixes/{NS}Prefix'):
            common.append(p.text)
        if root.find(f'{NS}IsTruncated').text != 'true':
            break
        nm = root.find(f'{NS}NextMarker')
        marker = (nm.text if nm is not None else (keys[-1] if keys else common[-1]))
    return keys, common


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------

def build_abide1_manifest() -> list[dict]:
    print('[manifest] listing ABIDE I func_preproc ...', flush=True)
    fp_keys, _ = list_bucket(ABIDE1_PREFIX + 'func_preproc/')
    print(f'[manifest] ABIDE I func_preproc files: {len(fp_keys)}', flush=True)
    out = []
    for k in fp_keys:
        fname = os.path.basename(k)
        if not fname.endswith('_func_preproc.nii.gz'):
            continue
        stem = fname[: -len('_func_preproc.nii.gz')]
        # stem = <SITE_LIKE>_<7digit_subid>, where SITE_LIKE may have underscores
        parts = stem.rsplit('_', 1)
        if len(parts) != 2:
            continue
        site, sub_id = parts
        mask_key = ABIDE1_PREFIX + 'func_mask/' + stem + '_func_mask.nii.gz'
        out.append({
            'subj': stem,           # e.g., NYU_0050952, LEUVEN_1_0050682
            'dataset': 'ABIDE_I',
            'site': site,
            'raw_id': sub_id,
            'bold_url': f'{S3}/{k}',
            'mask_url': f'{S3}/{mask_key}',
            'bold_filename': fname,
            'mask_filename': stem + '_func_mask.nii.gz',
        })
    return out


def build_abide2_manifest() -> list[dict]:
    print('[manifest] listing ABIDE II fmriprep ...', flush=True)
    _, sub_prefixes = list_bucket(ABIDE2_FMRIPREP_PREFIX, delimiter='/')
    sub_prefixes = [p for p in sub_prefixes if '/sub-' in p]
    print(f'[manifest] ABIDE II subject dirs: {len(sub_prefixes)}; '
          f'fetching per-subject keys in parallel ...', flush=True)

    def _list_sub(sp: str) -> list[str]:
        try:
            keys, _ = list_bucket(sp)
            return keys
        except Exception as e:
            print(f'[manifest] WARN list {sp}: {e}', flush=True)
            return []

    all_keys: list[str] = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        for keys in pool.map(_list_sub, sub_prefixes):
            all_keys.extend(keys)

    bolds = [k for k in all_keys if k.endswith('_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')]
    masks = set(k for k in all_keys if k.endswith('_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'))
    out = []
    for bk in bolds:
        mk = bk[: -len('_desc-preproc_bold.nii.gz')] + '_desc-brain_mask.nii.gz'
        if mk not in masks:
            continue
        fname = os.path.basename(bk)
        stem = fname[: -len('_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')]
        parts = stem.split('_')
        keep = [p for p in parts if p.startswith('sub-') or p.startswith('ses-') or p.startswith('run-')]
        subj = '_'.join(keep) if keep else stem
        sub_token = next((p for p in parts if p.startswith('sub-')), None)
        sub_id = sub_token[len('sub-'):] if sub_token else stem
        out.append({
            'subj': subj,
            'dataset': 'ABIDE_II',
            'site': '',  # filled in from phenotypic
            'raw_id': sub_id,
            'bold_url': f'{S3}/{bk}',
            'mask_url': f'{S3}/{mk}',
            'bold_filename': fname,
            'mask_filename': os.path.basename(mk),
        })
    print(f'[manifest] ABIDE II MNI bold runs: {len(out)}', flush=True)
    return out


# ---------------------------------------------------------------------------
# Phenotypic merging
# ---------------------------------------------------------------------------

def fetch_abide1_phenotypic() -> dict[str, dict]:
    """Return {SUB_ID -> {dataset, site, sub_id, dx_group, age, sex}}.

    Keyed by 7-digit padded SUB_ID since PCP filenames embed sub-site tags
    (e.g. CMU_a) that don't match the canonical SITE_ID column (CMU).
    """
    keys, _ = list_bucket(ABIDE1_PHENO_PREFIX)
    csvs = [k for k in keys if k.endswith('.csv') and 'phenotypic_' in k]
    rows: dict[str, dict] = {}
    for k in csvs:
        try:
            raw = _http_get(f'{S3}/{k}').decode('utf-8', errors='replace')
        except Exception as e:
            print(f'[phenotypic] WARN: {k}: {e}', flush=True)
            continue
        reader = csv.DictReader(io.StringIO(raw))
        for r in reader:
            site = (r.get('SITE_ID') or '').strip()
            sub_id = (r.get('SUB_ID') or '').strip()
            if not site or not sub_id:
                continue
            sub_id = sub_id.zfill(7)
            rows[sub_id] = {
                'dataset': 'ABIDE_I',
                'site': site,
                'sub_id': sub_id,
                'dx_group': (r.get('DX_GROUP') or '').strip(),  # 1=ASD, 2=TC
                'age': (r.get('AGE_AT_SCAN') or '').strip(),
                'sex': (r.get('SEX') or '').strip(),  # 1=M, 2=F
            }
    return rows


def fetch_abide2_phenotypic() -> dict[str, dict]:
    """Return {sub_id -> phenotypic dict}; ABIDE II site is encoded in TSV path."""
    _, site_prefixes = list_bucket(ABIDE2_RAW_PREFIX, delimiter='/')
    site_prefixes = [p for p in site_prefixes if '/ABIDEII-' in p]
    rows: dict[str, dict] = {}
    for sp in site_prefixes:
        site = sp.rstrip('/').rsplit('/', 1)[-1]  # ABIDEII-NYU_1
        url = f'{S3}/{sp}participants.tsv'
        try:
            raw = _http_get(url).decode('utf-8', errors='replace')
        except Exception:
            continue
        lines = raw.splitlines()
        if not lines:
            continue
        header = [h.strip() for h in lines[0].split('\t')]
        # find columns flexibly
        def col(name: str) -> int:
            for i, h in enumerate(header):
                if h.lower().strip() == name.lower():
                    return i
            return -1
        i_pid = col('participant_id')
        i_dx = col('dx_group')
        i_age = col('age_at_scan')
        i_sex = col('sex')
        i_site = col('site_id')
        if i_pid < 0:
            continue
        for ln in lines[1:]:
            cells = ln.split('\t')
            if len(cells) <= i_pid:
                continue
            pid = cells[i_pid].strip()
            if not pid:
                continue
            rows[pid] = {
                'subj': f'sub-{pid}',
                'dataset': 'ABIDE_II',
                'site': cells[i_site].strip() if i_site >= 0 and len(cells) > i_site else site,
                'sub_id': pid,
                'dx_group': cells[i_dx].strip() if i_dx >= 0 and len(cells) > i_dx else '',
                'age': cells[i_age].strip() if i_age >= 0 and len(cells) > i_age else '',
                'sex': cells[i_sex].strip() if i_sex >= 0 and len(cells) > i_sex else '',
            }
    return rows


def write_phenotypic_csv(save_root: str,
                         a1: dict[str, dict],
                         a2: dict[str, dict],
                         manifest: list[dict]) -> str:
    """Write merged phenotypic CSV; one row per subj entry in manifest."""
    out_path = os.path.join(save_root, 'metadata', 'phenotypic.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ['subj', 'dataset', 'site', 'sub_id', 'dx_group', 'age', 'sex',
                  'bold_filename']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in manifest:
            if m['dataset'] == 'ABIDE_I':
                # raw_id is already 7-digit padded from manifest builder
                p = a1.get(m['raw_id'], {})
            else:
                p = a2.get(m['raw_id'], {})
            w.writerow({
                'subj': m['subj'],
                'dataset': m['dataset'],
                'site': p.get('site', m.get('site', '')),
                'sub_id': p.get('sub_id', m['raw_id']),
                'dx_group': p.get('dx_group', ''),
                'age': p.get('age', ''),
                'sex': p.get('sex', ''),
                'bold_filename': m['bold_filename'],
            })
    return out_path


# ---------------------------------------------------------------------------
# Per-subject worker
# ---------------------------------------------------------------------------

# Globals populated by Pool initializer (atlases load once per worker)
_ATLASES: dict[str, np.ndarray] = {}
_FC_TYPES: tuple[str, ...] = DEFAULT_FC_TYPES


def _worker_init(atlas_names: list[str], fc_types: tuple[str, ...]):
    global _ATLASES, _FC_TYPES
    _ATLASES = {a: load_atlas(a) for a in atlas_names}
    _FC_TYPES = tuple(fc_types)


def _download_to(url: str, dest: str, retries: int = 5):
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return
    tmp = dest + '.part'
    last_err: Exception | None = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r, open(tmp, 'wb') as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest)
            return
        except Exception as e:
            last_err = e
            try:
                os.remove(tmp)
            except OSError:
                pass
            time.sleep(2 ** i)
    raise RuntimeError(f'download failed: {url} ({last_err})')


def _quantize_to_int8(data_global: torch.Tensor) -> tuple[torch.Tensor, float]:
    abs_max = float(data_global.abs().max().item())
    scale = abs_max / 127.0 if abs_max > 0 else 1.0
    data_int = (data_global / scale).round().clamp_(-127, 127).to(torch.int8)
    return data_int, scale


def _process_one(task: dict) -> tuple[str, str]:
    subj = task['subj']
    save_root = task['save_root']
    raw_dir = task['raw_dir']

    img_dir = os.path.join(save_root, 'img', subj)
    blob_path = os.path.join(img_dir, 'data.pt')

    # Decide what stages need to run (idempotent / resumable).
    need_voxel = not os.path.isfile(blob_path)
    need_atlas = []
    for atlas_name in _ATLASES.keys():
        roi_file = os.path.join(save_root, 'roi', atlas_name, f'{subj}_{atlas_name}.npy')
        fc_files = [
            os.path.join(save_root, 'fc', atlas_name, fc, f'{subj}_{atlas_name}_{fc}.npy')
            for fc in _FC_TYPES
        ]
        if not (os.path.isfile(roi_file) and all(os.path.isfile(f) for f in fc_files)):
            need_atlas.append(atlas_name)

    if not need_voxel and not need_atlas:
        return subj, 'skip_done'

    # IO: download bold + mask in parallel
    bold_path = os.path.join(raw_dir, task['bold_filename'])
    mask_path = os.path.join(raw_dir, task['mask_filename'])
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(_download_to, task['bold_url'], bold_path)
            f2 = pool.submit(_download_to, task['mask_url'], mask_path)
            f1.result(); f2.result()
    except Exception as e:
        return subj, f'error_download: {e}'

    try:
        # Load
        bold_img = nib.load(bold_path)
        bold_data = bold_img.get_fdata()
        bold_hdr = bold_img.header
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        mask_hdr = mask_img.header

        # Resample to 2mm + center crop 96^3 (skip temporal resampling for ABIDE:
        # TR=2s upsampled to 0.8s would 2.5x voxel storage with no new info; the
        # native TR is recorded in the manifest/phenotypic for downstream use.)
        bold_rs = spatial_resampling(bold_data, bold_hdr)
        bold_rs = select_middle_96(bold_rs)

        mask_rs = spatial_resampling(mask_data, mask_hdr)
        mask_rs = select_middle_96(mask_rs)
        background = mask_rs == 0

        bold_rs[background] = 0
        bold_rs[bold_rs < 0] = 0
        data = torch.from_numpy(bold_rs.astype(np.float32))

        # z-norm on foreground; fill background with min foreground
        fg = data[~background]
        global_mean = fg.mean()
        global_std = fg.std()
        data_temp = (data - global_mean) / global_std
        data_global = torch.empty_like(data)
        data_global[background] = data_temp[~background].min()
        data_global[~background] = data_temp[~background]

        # int8 quantize -> [T, H, W, D]
        data_int, scale = _quantize_to_int8(data_global)
        assert data_int.ndim == 4
        data_int = data_int.permute(3, 0, 1, 2).contiguous()

        if need_voxel:
            os.makedirs(img_dir, exist_ok=True)
            torch.save({
                'frames': data_int,
                'scale': float(scale),
                'num_frames': int(data_int.shape[0]),
            }, blob_path)

        # Volume for ROI/FC: dequantize so it matches what data.pt readers see
        if need_atlas:
            volume = (data_int.to(torch.float32) * float(scale)).numpy()  # [T, H, W, D]
            spatial_shape = volume.shape[1:4]
            for atlas_name in need_atlas:
                atlas_data = _ATLASES[atlas_name]
                atlas_resized = (atlas_data if atlas_data.shape == spatial_shape
                                 else resize_atlas_labels(atlas_data, spatial_shape))
                roi_dir = os.path.join(save_root, 'roi', atlas_name)
                roi_file = os.path.join(roi_dir, f'{subj}_{atlas_name}.npy')
                if os.path.isfile(roi_file):
                    roi_data = np.load(roi_file)
                else:
                    os.makedirs(roi_dir, exist_ok=True)
                    roi_data = extract_roi_from_volume(volume, atlas_resized)
                    np.save(roi_file, roi_data)
                for fc_type in _FC_TYPES:
                    fc_dir = os.path.join(save_root, 'fc', atlas_name, fc_type)
                    fc_file = os.path.join(fc_dir, f'{subj}_{atlas_name}_{fc_type}.npy')
                    if os.path.isfile(fc_file):
                        continue
                    os.makedirs(fc_dir, exist_ok=True)
                    fc = compute_fc(roi_data, fc_type)
                    np.save(fc_file, fc)

    except Exception as e:
        return subj, f'error_process: {e}'
    finally:
        for p in (bold_path, mask_path):
            try:
                os.remove(p)
            except OSError:
                pass

    return subj, 'ok'


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', required=True,
                        help='output dir, e.g. /data_ssd/cwang/fmri/abide_preprocessed')
    parser.add_argument('--raw_dir', required=True,
                        help='temp dir for downloaded NIfTIs (deleted per-subject)')
    parser.add_argument('--num_processes', type=int, default=6)
    parser.add_argument('--only', choices=['abide1', 'abide2', 'both'], default='both')
    parser.add_argument('--limit', type=int, default=0,
                        help='process only first N subjects (smoke test). 0 = all.')
    parser.add_argument('--atlases', nargs='+', default=None,
                        help='subset of atlases (default: all dirs in datasets/atlas/)')
    parser.add_argument('--fc_types', nargs='+', default=list(DEFAULT_FC_TYPES))
    parser.add_argument('--manifest_only', action='store_true',
                        help='build manifest + phenotypic CSV and exit')
    parser.add_argument('--refresh_manifest', action='store_true',
                        help='force rebuild of metadata/manifest.json')
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_root, 'metadata'), exist_ok=True)

    # Atlas list = every subdir in datasets/atlas/ unless overridden
    atlas_root = os.path.join(REPO_ROOT, 'datasets', 'atlas')
    all_atlases = sorted(d for d in os.listdir(atlas_root)
                         if os.path.isdir(os.path.join(atlas_root, d))
                         and os.path.exists(os.path.join(atlas_root, d, 'atlas.nii.gz'))
                         and len(nib.load(os.path.join(atlas_root, d, 'atlas.nii.gz')).shape) == 3)
    atlases = args.atlases if args.atlases else all_atlases
    print(f'[atlas] using {len(atlases)} atlases: {atlases}', flush=True)

    # Manifest (cache so reruns don't re-list S3)
    manifest_cache = os.path.join(args.save_root, 'metadata', 'manifest.json')
    if os.path.isfile(manifest_cache) and not args.refresh_manifest:
        import json
        with open(manifest_cache) as f:
            manifest = json.load(f)
        print(f'[manifest] loaded cache: {manifest_cache} ({len(manifest)} entries)', flush=True)
    else:
        manifest = []
        if args.only in ('abide1', 'both'):
            manifest.extend(build_abide1_manifest())
        if args.only in ('abide2', 'both'):
            manifest.extend(build_abide2_manifest())
        import json
        with open(manifest_cache, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f'[manifest] wrote cache: {manifest_cache}', flush=True)

    # Phenotypic
    a1_p = fetch_abide1_phenotypic() if args.only in ('abide1', 'both') else {}
    a2_p = fetch_abide2_phenotypic() if args.only in ('abide2', 'both') else {}
    pheno_csv = write_phenotypic_csv(args.save_root, a1_p, a2_p, manifest)
    print(f'[phenotypic] wrote {pheno_csv} (rows={len(manifest)}, '
          f'a1_phenotypic={len(a1_p)}, a2_phenotypic={len(a2_p)})', flush=True)

    if args.manifest_only:
        return

    if args.limit > 0:
        manifest = manifest[: args.limit]

    # Attach output paths to each task
    for m in manifest:
        m['save_root'] = args.save_root
        m['raw_dir'] = args.raw_dir

    print(f'[run] {len(manifest)} subjects, {args.num_processes} workers', flush=True)

    t0 = time.time()
    ok = 0; skip = 0; err = 0
    with Pool(processes=args.num_processes,
              initializer=_worker_init,
              initargs=(atlases, tuple(args.fc_types))) as pool:
        for subj, status in pool.imap_unordered(_process_one, manifest, chunksize=1):
            if status == 'ok':
                ok += 1
            elif status.startswith('skip'):
                skip += 1
            else:
                err += 1
                print(f'[err] {subj}: {status}', flush=True)
            done = ok + skip + err
            if done % 25 == 0 or done == len(manifest):
                rate = done / max(time.time() - t0, 1.0)
                eta = (len(manifest) - done) / max(rate, 1e-6)
                print(f'[progress] {done}/{len(manifest)} ok={ok} skip={skip} err={err} '
                      f'rate={rate:.2f}/s eta={eta/60:.1f}min', flush=True)


if __name__ == '__main__':
    main()





