#!/usr/bin/env python3
"""Cross-dataset experiments: 6 models x 2 tasks (sex, age) x N datasets.
Uses best (atlas, lr) per (model, task) found from prior HCP-YA sweeps.
"""
from __future__ import annotations
import csv
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PYTHON = "/home/cwang/anaconda3/envs/neurostorm/bin/python"
OUT_BASE = Path("output/cross_dataset")
RESULTS_CSV = OUT_BASE / "results.csv"

DATASETS = {
    "HCPD": "/home/cwang/remote/fmri/hcpd_preprocessed",
    "HCPA": "/home/cwang/remote/fmri/hcpa_processed",
    "ABCD": "/home/cwang/remote/fmri/abcd_preprocessed",
    "UKB": "/home/cwang/remote/fmri/ukb_preprocessed",
}

# Best (atlas, num_rois, fc_type, lr) per (model, task) from prior sweeps
BEST_CONFIGS = {
    # SEX
    ("brainnetcnn", "sex"): dict(atlas="glasser_360", num_rois=360, fc_type="correlation", lr=5e-4,
        data_type="fc_bnt", extra="--dropout 0.5 --e2e_channels 32 --e2n_channels 64 --n2g_channels 256 --weight_decay 5e-4 --optimizer Adam"),
    ("bnt", "sex"): dict(atlas="schaefer_400_7net", num_rois=400, fc_type="correlation", lr=1e-4,
        data_type="fc_bnt", extra="--dropout 0.1 --pos_embed_dim 8 --hidden_size 1024 --weight_decay 1e-4 --optimizer Adam"),
    ("braingnn", "sex"): dict(atlas="destrieux_148", num_rois=148, fc_type="correlation", lr=1e-3,
        data_type="fc_graph", extra="--dropout 0.5 --pooling_ratio 0.5 --num_communities 16 --weight_decay 1e-3 --optimizer SGD --momentum 0.9"),
    ("combraintf", "sex"): dict(atlas="basc_122", num_rois=122, fc_type="correlation", lr=5e-4,
        data_type="fc_bnt", extra="--dropout 0.1 --d_model 128 --nhead 4 --num_layers 3 --dim_feedforward 512 --num_communities 10 --weight_decay 1e-4 --optimizer Adam"),
    ("ibgnn", "sex"): dict(atlas="power_264", num_rois=255, fc_type="correlation", lr=5e-4,
        data_type="fc_graph", extra="--dropout 0.5 --hidden_dims 128 64 --weight_decay 5e-4 --optimizer Adam"),
    ("lggnn", "sex"): dict(atlas="glasser_360", num_rois=360, fc_type="correlation", lr=1e-3,
        data_type="fc_bnt", extra="--dropout 0.5 --hidden_dims 128 64 --k_neighbors 10 --learn_graph True --graph_metric cosine --weight_decay 5e-4 --optimizer Adam"),
    # AGE
    ("brainnetcnn", "age"): dict(atlas="cc200", num_rois=190, fc_type="correlation", lr=1e-4,
        data_type="fc_bnt", extra="--dropout 0.5 --e2e_channels 32 --e2n_channels 64 --n2g_channels 256 --weight_decay 5e-4 --optimizer Adam --label_scaling_method standardization"),
    ("bnt", "age"): dict(atlas="basc_122", num_rois=122, fc_type="correlation", lr=1e-4,
        data_type="fc_bnt", extra="--dropout 0.1 --pos_embed_dim 8 --hidden_size 1024 --weight_decay 1e-4 --optimizer Adam --label_scaling_method standardization"),
    ("braingnn", "age"): dict(atlas="schaefer_100_7net", num_rois=100, fc_type="partial_correlation", lr=1e-3,
        data_type="fc_graph", extra="--dropout 0.5 --pooling_ratio 0.5 --num_communities 16 --weight_decay 1e-3 --optimizer SGD --momentum 0.9 --label_scaling_method standardization"),
    ("combraintf", "age"): dict(atlas="cc200", num_rois=190, fc_type="correlation", lr=5e-4,
        data_type="fc_bnt", extra="--dropout 0.1 --d_model 128 --nhead 4 --num_layers 3 --dim_feedforward 512 --num_communities 10 --weight_decay 1e-4 --optimizer Adam --label_scaling_method standardization"),
    ("ibgnn", "age"): dict(atlas="cc200", num_rois=190, fc_type="partial_correlation", lr=5e-4,
        data_type="fc_graph", extra="--dropout 0.5 --hidden_dims 128 64 --weight_decay 5e-4 --optimizer Adam --label_scaling_method standardization"),
    ("lggnn", "age"): dict(atlas="cc200", num_rois=190, fc_type="correlation", lr=1e-3,
        data_type="fc_bnt", extra="--dropout 0.5 --hidden_dims 128 64 --k_neighbors 10 --learn_graph True --graph_metric cosine --weight_decay 5e-4 --optimizer Adam --label_scaling_method standardization"),
}

# data_type for label
def task_kwargs(task):
    if task == "sex":
        return "--downstream_task_type classification --task_name sex --num_classes 2"
    return "--downstream_task_type regression --task_name age --num_classes 1"


GPUS = [4, 5, 6, 7]
MAX_EPOCHS = 30


def build_jobs(datasets):
    jobs = []
    for ds in datasets:
        for (model, task), cfg in BEST_CONFIGS.items():
            tag = f"{model}_{ds.lower()}_{task}"
            jobs.append(dict(
                tag=tag, dataset=ds, image_path=DATASETS[ds],
                model=model, task=task,
                **cfg,
            ))
    return jobs


def run_job(job, gpu):
    out_dir = OUT_BASE / job["tag"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir.with_suffix(".log")
    if (out_dir / "_done").exists():
        return job, _read_metric(log_path, job["task"]), gpu

    # Auto-fallback: if partial_correlation/atlas not available for this dataset,
    # try correlation; if atlas not present either, skip with None.
    fc_dir = Path(job["image_path"]) / "fc" / job["atlas"] / job["fc_type"]
    if not fc_dir.is_dir() or not any(fc_dir.iterdir()):
        # Try correlation fallback
        alt = Path(job["image_path"]) / "fc" / job["atlas"] / "correlation"
        if alt.is_dir() and any(alt.iterdir()):
            print(f"  [fallback] {job['tag']}: {job['fc_type']} -> correlation")
            job = {**job, "fc_type": "correlation"}
        else:
            # Try a different atlas with correlation: cc200 if available
            for fb_atlas, fb_n in [("cc200", 190), ("schaefer_100_7net", 100), ("aal_116", 116)]:
                alt = Path(job["image_path"]) / "fc" / fb_atlas / "correlation"
                if alt.is_dir() and any(alt.iterdir()):
                    print(f"  [fallback] {job['tag']}: {job['atlas']}/{job['fc_type']} -> {fb_atlas}/correlation")
                    job = {**job, "atlas": fb_atlas, "num_rois": fb_n, "fc_type": "correlation"}
                    break
            else:
                print(f"  [skip] {job['tag']}: no FC available")
                return job, None, gpu

    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu} {PYTHON} main.py "
        f"--output_dir {out_dir} --project_name {job['tag']} --loggername tensorboard "
        f"--dataset_name {job['dataset']} --image_path {job['image_path']} "
        f"--num_nodes 1 --seed 1234 "
        f"--model {job['model']} --data_type {job['data_type']} "
        f"--atlas_name {job['atlas']} --fc_type {job['fc_type']} --num_rois {job['num_rois']} "
        f"--downstream_task_id 1 {task_kwargs(job['task'])} "
        f"--learning_rate {job['lr']} {job['extra']} "
        f"--batch_size 16 --eval_batch_size 32 --max_epochs {MAX_EPOCHS} --num_workers 4"
    )
    with open(log_path, "w") as f:
        ret = subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    if ret == 0:
        (out_dir / "_done").touch()
    return job, _read_metric(log_path, job["task"]), gpu


def _read_metric(log_path, task):
    try:
        text = log_path.read_text()
        key = "test_acc" if task == "sex" else "test_mae"
        m = re.findall(rf"{key}\s+│\s+([0-9.e+-]+)", text)
        if m:
            return float(m[-1])
    except FileNotFoundError:
        pass
    return None


def main():
    import sys
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    datasets = sys.argv[1:] or ["HCPD", "HCPA", "ABCD"]
    print(f"Datasets: {datasets}")
    jobs = build_jobs(datasets)
    print(f"Total jobs: {len(jobs)}")

    if not RESULTS_CSV.exists():
        with open(RESULTS_CSV, "w", newline="") as f:
            csv.writer(f).writerow(
                ["dataset", "model", "task", "atlas", "num_rois", "fc_type", "lr", "metric", "tag"]
            )

    gpu_queue = list(GPUS)
    pending = list(jobs)
    futures = {}
    with ProcessPoolExecutor(max_workers=len(GPUS)) as ex:
        while pending and gpu_queue:
            job = pending.pop(0)
            gpu = gpu_queue.pop(0)
            futures[ex.submit(run_job, job, gpu)] = job

        while futures:
            done = next(as_completed(futures))
            job = futures.pop(done)
            try:
                j, metric, gpu = done.result()
            except Exception as e:
                print(f"  FAIL {job['tag']}: {e}")
                gpu = GPUS[0]; metric = None; j = job
            print(f"  {j['tag']:<55} metric={metric} (gpu {gpu})")
            with open(RESULTS_CSV, "a", newline="") as f:
                csv.writer(f).writerow([
                    j["dataset"], j["model"], j["task"], j["atlas"],
                    j["num_rois"], j["fc_type"], j["lr"],
                    metric if metric is not None else "", j["tag"],
                ])
            gpu_queue.append(gpu)
            if pending:
                next_job = pending.pop(0)
                next_gpu = gpu_queue.pop(0)
                futures[ex.submit(run_job, next_job, next_gpu)] = next_job


if __name__ == "__main__":
    main()
