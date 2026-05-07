#!/bin/bash

# ==================== CONFIGURATION ====================
MODEL="bbdrec"
DATASET="yelp"

NUM_GPUS=4              # 使用的 GPU 数量
TASKS_PER_GPU=1         # 每张卡同时跑的任务数（防 OOM）
GPU_START=0             # 起始 GPU 编号 (cuda:0, cuda:1, ...)

# ==================== SETUP ====================
RESULT_DIR="logs/${MODEL}_${DATASET}"
mkdir -p "$RESULT_DIR"

exec > "${RESULT_DIR}/tune_bbdrec.log" 2>&1

DS_LIST=(2 4 8 16 32)
VM_LIST=(0.5 1 2 4 8)
LS_LIST=(0.01 0.1 1 10)

TOTAL=$(( ${#DS_LIST[@]} * ${#VM_LIST[@]} * ${#LS_LIST[@]} ))
MAX_CONCURRENT=$(( NUM_GPUS * TASKS_PER_GPU ))

echo "============================================"
echo "  BBDRec Tuning: ${TOTAL} combos  |  Dataset: ${DATASET}"
echo "  GPUs: ${NUM_GPUS} (cuda:${GPU_START} ~ cuda:$((GPU_START+NUM_GPUS-1)))"
echo "  Tasks/GPU: ${TASKS_PER_GPU}  |  Max concurrent: ${MAX_CONCURRENT}"
echo "  Mode: GPU Workers (per-GPU limit enforced)"
echo "============================================"

# ---- 1. Build per-GPU task files (round-robin assignment) ----
task_id=0
for ((g=0; g<NUM_GPUS; g++)); do
    > "${RESULT_DIR}/.gpu_${g}_tasks.txt"
done

for ds in "${DS_LIST[@]}"; do
    for vm in "${VM_LIST[@]}"; do
        for ls in "${LS_LIST[@]}"; do
            g=$(( task_id % NUM_GPUS ))
            gpu="cuda:$((GPU_START + g))"
            desc="ds${ds}_vm${vm}_ls${ls}"
            log_path="${RESULT_DIR}/${MODEL}_${DATASET}_${desc}.log"
            # Format: ds|vm|ls|gpu|desc|log_path|global_task_id
            echo "${ds}|${vm}|${ls}|${gpu}|${desc}|${log_path}|${task_id}" \
                >> "${RESULT_DIR}/.gpu_${g}_tasks.txt"
            task_id=$((task_id + 1))
        done
    done
done

echo "Tasks distributed: ${TOTAL} total across ${NUM_GPUS} GPUs"
echo ""

# ---- 2. Run GPU workers (each GPU independently limited to TASKS_PER_GPU) ----
for ((g=0; g<NUM_GPUS; g++)); do
    gpu="cuda:$((GPU_START + g))"
    task_file="${RESULT_DIR}/.gpu_${g}_tasks.txt"

    # One background subshell per GPU
    (
        running=0
        gpu_completed=0

        while IFS='|' read -r ds vm ls gpu2 desc log_path tid; do

            # ---- throttle: wait until this GPU has a free slot ----
            while [ $running -ge $TASKS_PER_GPU ]; do
                wait -n 2>/dev/null
                running=$((running - 1))
                gpu_completed=$((gpu_completed + 1))
            done

            echo "[$(date '+%H:%M:%S')] START  [#${tid}]  ds=${ds}  vm=${vm}  ls=${ls}  ->  ${gpu}"

            # ---- launch one task on this GPU ----
            (
                python -u main.py \
                    --dataset "${DATASET}" \
                    --model "${MODEL}" \
                    --diffusion_steps "${ds}" \
                    --var_max "${vm}" \
                    --loss_scale "${ls}" \
                    --device "${gpu}" \
                    --description "${desc}" \
                    > "${log_path}" 2>&1
                exit_code=$?
                echo "[$(date '+%H:%M:%S')] DONE   [#${tid}]  ds=${ds}  vm=${vm}  ls=${ls}  (exit=${exit_code})"
            ) &

            running=$((running + 1))

        done < "${task_file}"

        # ---- drain remaining tasks on this GPU ----
        while [ $running -gt 0 ]; do
            wait -n 2>/dev/null
            running=$((running - 1))
            gpu_completed=$((gpu_completed + 1))
        done

        echo "[GPU ${gpu}] all ${gpu_completed} tasks finished."

    ) &

done

# ---- Wait for all GPU workers to finish ----
wait

# ---- Cleanup temp files ----
rm -f "${RESULT_DIR}"/.gpu_*.txt

echo ""
echo "All ${TOTAL} experiments finished!"

# ---- 3. Extract results ----
echo ""
echo "============================================"
echo "  Results (test HR@20 / NDCG@20)"
echo "============================================"

> "${RESULT_DIR}/results.txt"

for ds in "${DS_LIST[@]}"; do
    for vm in "${VM_LIST[@]}"; do
        for ls in "${LS_LIST[@]}"; do
            desc="ds${ds}_vm${vm}_ls${ls}"
            log="${RESULT_DIR}/${MODEL}_${DATASET}_${desc}.log"

            test_line=$(grep "^Test:" "${log}" 2>/dev/null | tail -1)

            if [ -n "${test_line}" ]; then
                hr20=$(echo "${test_line}" | grep -oP "'HR@20':\s*\K[0-9.]+")
                ndcg20=$(echo "${test_line}" | grep -oP "'NDCG@20':\s*\K[0-9.]+")
                [ -z "${hr20}" ] && hr20="N/A"
                [ -z "${ndcg20}" ] && ndcg20="N/A"
                printf "[OK]    ds=%-2s  vm=%-4s  ls=%-5s  |  HR@20=%-8s  NDCG@20=%-8s\n" \
                       "${ds}" "${vm}" "${ls}" "${hr20}" "${ndcg20}"
                echo "${hr20} ${ndcg20} ds=${ds} vm=${vm} ls=${ls}" >> "${RESULT_DIR}/results.txt"
            else
                printf "[FAIL]  ds=%-2s  vm=%-4s  ls=%-5s\n" "${ds}" "${vm}" "${ls}"
            fi
        done
    done
done

# ---- 4. Best combo ----
echo ""
echo "============================================"
echo "  Best Hyperparameters (by HR@20)"
echo "============================================"

if [ ! -s "${RESULT_DIR}/results.txt" ]; then
    echo "No successful experiments!"
    exit 1
fi

best=$(sort -k1,1nr "${RESULT_DIR}/results.txt" | head -1)
read -r hr20 ndcg20 rest <<< "${best}"
echo ""
echo "  ${rest}"
echo "  HR@20   = ${hr20}"
echo "  NDCG@20 = ${ndcg20}"
echo ""
echo "============================================"
echo "  Done!"
echo "============================================"
