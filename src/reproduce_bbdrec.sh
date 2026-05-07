#!/bin/bash

# ==================== CONFIGURATION ====================
# Reproduce BBDRec best hyperparameters for each dataset
# Best settings determined from hyperparameter tuning logs

MODEL="bbdrec"
DATASETS=("baby" "beauty" "ml-100k" "sports" "toys" "yelp")

NUM_GPUS=3              # 使用的 GPU 数量
TASKS_PER_GPU=2         # 每张卡同时跑的任务数（防 OOM）
GPU_START=5             # 起始 GPU 编号 (cuda:0, cuda:1, ...)

# ==================== BEST HYPERPARAMETERS ====================
# Format: dataset ds vm ls
declare -A BEST_PARAMS
BEST_PARAMS["baby"]="2 4 1"
BEST_PARAMS["beauty"]="8 0.5 1"
BEST_PARAMS["ml-100k"]="4 0.5 10"
BEST_PARAMS["sports"]="2 0.5 0.01"
BEST_PARAMS["toys"]="16 4 1"
BEST_PARAMS["yelp"]="2 4 0.1"

# ==================== SETUP ====================
RESULT_DIR="logs/reproduce_bbdrec"
mkdir -p "$RESULT_DIR"

exec > "${RESULT_DIR}/reproduce_bbdrec.log" 2>&1

TOTAL=${#DATASETS[@]}
MAX_CONCURRENT=$(( NUM_GPUS * TASKS_PER_GPU ))

echo "============================================"
echo "  BBDRec Best Reproduction"
echo "  Model: ${MODEL}"
echo "  Datasets: ${DATASETS[*]}"
echo "  Total: ${TOTAL} experiments"
echo "  GPUs: ${NUM_GPUS} (cuda:${GPU_START} ~ cuda:$((GPU_START+NUM_GPUS-1)))"
echo "  Tasks/GPU: ${TASKS_PER_GPU}  |  Max concurrent: ${MAX_CONCURRENT}"
echo "============================================"
echo ""

# ---- Print best hyperparams ----
echo "Best Hyperparameters:"
for dataset in "${DATASETS[@]}"; do
    read -r best_ds best_vm best_ls <<< "${BEST_PARAMS[$dataset]}"
    echo "  ${dataset}: ds=${best_ds}, vm=${best_vm}, ls=${best_ls}"
done
echo ""

# ---- 1. Build per-GPU task files (round-robin assignment) ----
task_id=0
for ((g=0; g<NUM_GPUS; g++)); do
    > "${RESULT_DIR}/.gpu_${g}_tasks.txt"
done

for dataset in "${DATASETS[@]}"; do
    read -r best_ds best_vm best_ls <<< "${BEST_PARAMS[$dataset]}"
    
    g=$(( task_id % NUM_GPUS ))
    gpu="cuda:$((GPU_START + g))"
    desc="${MODEL}_${dataset}_best"
    log_path="${RESULT_DIR}/${MODEL}_${dataset}.log"
    
    # Format: dataset|ds|vm|ls|gpu|desc|log_path|task_id
    echo "${dataset}|${best_ds}|${best_vm}|${best_ls}|${gpu}|${desc}|${log_path}|${task_id}" \
        >> "${RESULT_DIR}/.gpu_${g}_tasks.txt"
    
    task_id=$((task_id + 1))
done

echo "Tasks distributed: ${TOTAL} total across ${NUM_GPUS} GPUs"
echo ""

# ---- 2. Run GPU workers ----
for ((g=0; g<NUM_GPUS; g++)); do
    gpu="cuda:$((GPU_START + g))"
    task_file="${RESULT_DIR}/.gpu_${g}_tasks.txt"

    (
        running=0
        gpu_completed=0

        while IFS='|' read -r dataset ds vm ls gpu2 desc log_path tid; do

            # throttle: wait until this GPU has a free slot
            while [ $running -ge $TASKS_PER_GPU ]; do
                wait -n 2>/dev/null
                running=$((running - 1))
                gpu_completed=$((gpu_completed + 1))
            done

            echo "[$(date '+%H:%M:%S')] START  [#${tid}]  dataset=${dataset}  ds=${ds}  vm=${vm}  ls=${ls}  ->  ${gpu}"

            (
                python -u main.py \
                    --dataset "${dataset}" \
                    --model "${MODEL}" \
                    --diffusion_steps "${ds}" \
                    --var_max "${vm}" \
                    --loss_scale "${ls}" \
                    --device "${gpu}" \
                    --description "${desc}" \
                    > "${log_path}" 2>&1
                exit_code=$?
                echo "[$(date '+%H:%M:%S')] DONE   [#${tid}]  dataset=${dataset}  ds=${ds}  vm=${vm}  ls=${ls}  (exit=${exit_code})"
            ) &

            running=$((running + 1))

        done < "${task_file}"

        # drain remaining tasks on this GPU
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

echo ""
printf "%-10s %-6s %-6s %-6s %-10s %-10s\n" "Dataset" "ds" "vm" "ls" "HR@20" "NDCG@20"
echo "---------------------------------------------------------------"

for dataset in "${DATASETS[@]}"; do
    read -r best_ds best_vm best_ls <<< "${BEST_PARAMS[$dataset]}"
    log="${RESULT_DIR}/${MODEL}_${dataset}.log"
    
    test_line=$(grep "^Test:" "${log}" 2>/dev/null | tail -1)
    
    if [ -n "${test_line}" ]; then
        hr20=$(echo "${test_line}" | grep -oP "'HR@20':\s*\K[0-9.]+")
        ndcg20=$(echo "${test_line}" | grep -oP "'NDCG@20':\s*\K[0-9.]+")
        [ -z "${hr20}" ] && hr20="N/A"
        [ -z "${ndcg20}" ] && ndcg20="N/A"
        printf "%-10s %-6s %-6s %-6s %-10s %-10s\n" \
               "${dataset}" "${best_ds}" "${best_vm}" "${best_ls}" "${hr20}" "${ndcg20}"
        echo "${hr20} ${ndcg20} ${dataset} ds=${best_ds} vm=${best_vm} ls=${best_ls}" >> "${RESULT_DIR}/results.txt"
    else
        printf "%-10s %-6s %-6s %-6s %-10s %-10s\n" \
               "${dataset}" "${best_ds}" "${best_vm}" "${best_ls}" "FAIL" "FAIL"
    fi
done

echo ""
echo "============================================"
echo "  Done!"
echo "============================================"
