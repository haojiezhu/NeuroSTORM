#!/usr/bin/env python3
"""Atlas + LR sweep for sex classification on HCP-YA.

For each (model, atlas, lr) combo:
  - Train 30 epochs
  - Save test_acc to a results CSV

Runs N processes at a time across the given GPUs.
"""
from __future__ import annotations
import csv
import os
import re
import shlex
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PYTHON = "/home/cwang/anaconda3/envs/neurostorm/bin/python"
DATA_PATH = "/data/cwang/remote/fmri/hcpya_preprocessed"
OUT_BASE = Path("output/sex_sweep")
RESULTS_CSV = OUT_BASE / "results.csv"

# atlas -> (num_rois, fc_type for graph models)
ATLASES = {
    "aal_116":              (116, "correlation"),
    "aal3_166":              (166, "correlation"),
    "basc_122":              (122, "correlation"),
    "cc200":                (190, "correlation"),
    "destrieux_148":        (148, "correlation"),
    "dk_112":                (112, "correlation"),
    "glasser_360":          (360, "correlation"),
    "harvard_oxford_cort":  (48,  "correlation"),
    "harvard_oxford_sub":   (21,  "correlation"),
    "power_264":            (255, "correlation"),
    "schaefer_100_7net":    (100, "correlation"),
    "schaefer_200_7net":    (200, "correlation"),
    "schaefer_400_7net":    (400, "correlation"),
}

# graph models that need partial_correlation (fall back to correlation if unavailable)
GRAPH_PARTIAL_OK = {"cc200", "schaefer_100_7net"}

MODELS = {
    "brainnetcnn": dict(
        data_type="fc_bnt", lrs=[1e-3, 5e-4],
        extra="--dropout 0.5 --e2e_channels 32 --e2n_channels 64 --n2g_channels 256 --weight_decay 5e-4",
    ),
    "bnt": dict(
        data_type="fc_bnt", lrs=[1e-4, 5e-5],
        extra="--dropout 0.1 --pos_embed_dim 8 --hidden_size 1024 --weight_decay 1e-4",
    ),
    "braingnn": dict(
        data_type="fc_graph", lrs=[1e-3, 5e-4],
        extra="--dropout 0.5 --pooling_ratio 0.5 --num_communities 16 --weight_decay 1e-3 --optimizer SGD --momentum 0.9",
    ),
    "combraintf": dict(
        data_type="fc_bnt", lrs=[5e-4, 1e-4],
        extra="--dropout 0.1 --d_model 128 --nhead 4 --num_layers 3 --dim_feedforward 512 --num_communities 10 --weight_decay 1e-4",
    ),
    "ibgnn": dict(
        data_type="fc_graph", lrs=[5e-4, 1e-4],
        extra="--dropout 0.5 --hidden_dims 128 64 --weight_decay 5e-4",
    ),
    "lggnn": dict(
        data_type="fc_bnt", lrs=[1e-3, 5e-4],
        extra="--dropout 0.5 --hidden_dims 128 64 --k_neighbors 10 --learn_graph True --graph_metric cosine --weight_decay 5e-4",
    ),
}

GPUS = [4, 5, 6, 7]
MAX_EPOCHS = 30


def build_jobs():
    jobs = []
    for model, cfg in MODELS.items():
        for atlas, (nrois, _) in ATLASES.items():
            fc_type = "correlation"
            if cfg["data_type"] == "fc_graph" and atlas in GRAPH_PARTIAL_OK:
                fc_type = "partial_correlation"
            for lr in cfg["lrs"]:
                tag = f"{model}_{atlas}_lr{lr:.0e}".replace("-0", "-")
                jobs.append(dict(
                    model=model, atlas=atlas, num_rois=nrois,
                    fc_type=fc_type, data_type=cfg["data_type"],
                    lr=lr, extra=cfg["extra"], tag=tag,
                ))
    return jobs


def run_job(job, gpu):
    out_dir = OUT_BASE / job["tag"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir.with_suffix(".log")
    if (out_dir / "_done").exists():
        return job, _read_acc(log_path), gpu

    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu} {PYTHON} main.py "
        f"--output_dir {out_dir} --project_name {job['tag']} --loggername tensorboard "
        f"--dataset_name HCP1200 --image_path {DATA_PATH} --num_nodes 1 --seed 1234 "
        f"--model {job['model']} --data_type {job['data_type']} "
        f"--atlas_name {job['atlas']} --fc_type {job['fc_type']} --num_rois {job['num_rois']} "
        f"--downstream_task_id 1 --downstream_task_type classification --task_name sex --num_classes 2 "
        f"--learning_rate {job['lr']} {job['extra']} "
        f"--batch_size 16 --eval_batch_size 32 --max_epochs {MAX_EPOCHS} --num_workers 4"
    )
    if "optimizer" not in job["extra"]:
        cmd += " --optimizer Adam"
    with open(log_path, "w") as f:
        ret = subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    if ret == 0:
        (out_dir / "_done").touch()
    return job, _read_acc(log_path), gpu


def _read_acc(log_path):
    try:
        text = log_path.read_text()
        m = re.findall(r"test_acc\s+│\s+([0-9.]+)", text)
        if m:
            return float(m[-1])
    except FileNotFoundError:
        pass
    return None


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    print(f"Total jobs: {len(jobs)}")

    # write CSV header
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "atlas", "num_rois", "fc_type", "lr", "test_acc", "tag"])

    # GPU pool: workers = len(GPUS), cycle gpu assignment
    gpu_queue = list(GPUS)
    pending = list(jobs)
    futures = {}

    with ProcessPoolExecutor(max_workers=len(GPUS)) as ex:
        # initial fill
        while pending and gpu_queue:
            job = pending.pop(0)
            gpu = gpu_queue.pop(0)
            fut = ex.submit(run_job, job, gpu)
            futures[fut] = job

        while futures:
            done = next(as_completed(futures))
            job = futures.pop(done)
            try:
                j, acc, gpu = done.result()
            except Exception as e:
                print(f"  FAIL {job['tag']}: {e}")
                gpu = GPUS[0]
                acc = None
                j = job
            print(f"  {j['tag']:<60} acc={acc} (gpu {gpu})")
            with open(RESULTS_CSV, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([j["model"], j["atlas"], j["num_rois"], j["fc_type"],
                            j["lr"], acc if acc is not None else "", j["tag"]])
            # release GPU
            gpu_queue.append(gpu)
            # schedule next
            if pending:
                next_job = pending.pop(0)
                next_gpu = gpu_queue.pop(0)
                fut = ex.submit(run_job, next_job, next_gpu)
                futures[fut] = next_job


if __name__ == "__main__":
    main()
