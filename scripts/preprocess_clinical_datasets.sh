#!/bin/bash
# Preprocess clinical datasets: blob conversion + ROI/FC for all atlases.
# Reads from nas/<DATASET>_MNI_to_TRs_minmax, writes to nas/<dataset>_preprocessed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAS_DIR="${ROOT_DIR}/nas"
LOG_DIR="${ROOT_DIR}/output/preprocess_logs"
mkdir -p "${LOG_DIR}"

NUM_PROC_BLOB="${NUM_PROC_BLOB:-16}"
NUM_PROC_FC="${NUM_PROC_FC:-12}"

ATLASES=(
    aal_116 aal3_166 basc_122 cc200 cc400
    destrieux_148 dk_112 dosenbach_160 eickhoff_zilles
    glasser_360 harvard_oxford_cort harvard_oxford_merged harvard_oxford_sub
    power_264 schaefer_100_7net schaefer_200_7net schaefer_400_7net
    talairach_tournoux
)
FC_TYPES=(correlation partial_correlation)

# Map: SRC_NAME -> dst_name
declare -A DATASETS=(
    [ADHD200_MNI_to_TRs_minmax]=adhd200_preprocessed
    [COBRE_MNI_to_TRs_minmax]=cobre_preprocessed
    [HCPEP_MNI_to_TRs_minmax]=hcpep_preprocessed
    [UCLA_MNI_to_TRs_minmax]=ucla_preprocessed
)

for SRC in ADHD200_MNI_to_TRs_minmax COBRE_MNI_to_TRs_minmax HCPEP_MNI_to_TRs_minmax UCLA_MNI_to_TRs_minmax; do
    DST="${DATASETS[$SRC]}"
    SRC_DIR="${NAS_DIR}/${SRC}"
    DST_DIR="${NAS_DIR}/${DST}"
    LOG="${LOG_DIR}/${DST}.log"

    echo "==============================================" | tee -a "${LOG}"
    echo " ${SRC} -> ${DST}  ($(date -Iseconds))"        | tee -a "${LOG}"
    echo "==============================================" | tee -a "${LOG}"

    mkdir -p "${DST_DIR}/img"

    # Step 1: per-frame .pt -> int8 blob data.pt
    echo "[step 1/3] blob conversion (procs=${NUM_PROC_BLOB})" | tee -a "${LOG}"
    python "${ROOT_DIR}/datasets/convert_frames_to_blob.py" \
        --input_dir  "${SRC_DIR}/img" \
        --output_dir "${DST_DIR}/img" \
        --num_processes "${NUM_PROC_BLOB}" 2>&1 | tee -a "${LOG}"

    # Step 2: ROI + FC across all atlases
    echo "[step 2/3] ROI + FC (procs=${NUM_PROC_FC})" | tee -a "${LOG}"
    python "${ROOT_DIR}/datasets/compute_roi_fc.py" \
        --input_dir  "${DST_DIR}/img" \
        --input_format blob \
        --output_dir "${DST_DIR}" \
        --atlas_names "${ATLASES[@]}" \
        --fc_types "${FC_TYPES[@]}" \
        --num_processes "${NUM_PROC_FC}" 2>&1 | tee -a "${LOG}"

    # Step 3: copy metadata
    echo "[step 3/3] copy metadata" | tee -a "${LOG}"
    if [[ -d "${SRC_DIR}/metadata" ]]; then
        mkdir -p "${DST_DIR}/metadata"
        cp -rn "${SRC_DIR}/metadata/." "${DST_DIR}/metadata/" 2>&1 | tee -a "${LOG}"
    fi

    echo "[done] ${DST}  ($(date -Iseconds))" | tee -a "${LOG}"
done

echo "ALL DONE  ($(date -Iseconds))"
