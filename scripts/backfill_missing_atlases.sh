#!/bin/bash
# Backfill missing atlases for already-preprocessed datasets.
# Existing ROI/FC files are skipped automatically by compute_roi_fc.py.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAS_DIR="${ROOT_DIR}/nas"
LOG_DIR="${ROOT_DIR}/output/preprocess_logs"
mkdir -p "${LOG_DIR}"

NUM_PROC_FC="${NUM_PROC_FC:-16}"

ATLASES=(
    aal_116 aal3_166 basc_122 cc200 cc400
    destrieux_148 dk_112 dosenbach_160 eickhoff_zilles
    glasser_360 harvard_oxford_cort harvard_oxford_merged harvard_oxford_sub
    power_264 schaefer_100_7net schaefer_200_7net schaefer_400_7net
    talairach_tournoux
)
FC_TYPES=(correlation partial_correlation)

# Datasets that need backfill — order by ascending workload so quick wins land first.
DATASETS=(
    transdiag_preprocessed
    hcpd_preprocessed
    hcpa_processed
    hcpya_preprocessed
    abcd_preprocessed
    ukb_preprocessed
)

for DST in "${DATASETS[@]}"; do
    DST_DIR="${NAS_DIR}/${DST}"
    LOG="${LOG_DIR}/${DST}_backfill.log"

    echo "==============================================" | tee -a "${LOG}"
    echo " Backfill ${DST}  ($(date -Iseconds))"          | tee -a "${LOG}"
    echo "==============================================" | tee -a "${LOG}"

    python "${ROOT_DIR}/datasets/compute_roi_fc.py" \
        --input_dir  "${DST_DIR}/img" \
        --input_format blob \
        --output_dir "${DST_DIR}" \
        --atlas_names "${ATLASES[@]}" \
        --fc_types "${FC_TYPES[@]}" \
        --num_processes "${NUM_PROC_FC}" 2>&1 | tee -a "${LOG}"

    echo "[done] ${DST}  ($(date -Iseconds))" | tee -a "${LOG}"
done

echo "ALL BACKFILL DONE  ($(date -Iseconds))"
