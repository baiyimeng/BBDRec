#!/bin/bash

# ==================== CONFIGURATION ====================
MODELS=("diffurec" "dreamrec")
DATASETS=("baby" "beauty" "ml-100k" "sports" "toys" "yelp")

NUM_GPUS=4             # 使用的 GPU 数量
TASKS_PER_GPU=3         # 每张卡同时跑的任务数
GPU_START=0             # 起始 GPU 编号 (cuda:0, cuda:1, ...)

# ==================== SETUP ====================
RESULT_DIR="logs/baselines"
mkdir -p "$RESULT_DIR"

exec > "${RESULT_DIR}/run_baselines.log" 2>&1

TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))
MAX_CONCURRENT=$(( NUM_GPUS * TASKS_PER_GPU ))

echo "============================================"
echo "  Baselines: ${MODELS[*]}"
echo "  Datasets:  ${DATASETS[*]}"
echo "  Total: ${TOTAL} experiments"
echo "  GPUs: ${NUM_GPUS} (cuda:${GPU_START} ~ cuda:$((GPU_START+NUM_GPUS-1)))"
echo "  Tasks/GPU: ${TASKS_PER_GPU}  |  Max concurrent: ${MAX_CONCURRENT}"
echo "============================================"

# ---- 1. Build per-GPU task files (round-robin assignment) ----
task_id=0
for ((g=0; g<NUM_GPUS; g++)); do
    > "${RESULT_DIR}/.gpu_${g}_tasks.txt"
done

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        g=$(( task_id % NUM_GPUS ))
        gpu="cuda:$((GPU_START + g))"
        desc="${model}_${dataset}"
        log_path="${RESULT_DIR}/${model}_${dataset}.log"
        echo "${model}|${dataset}|${gpu}|${desc}|${log_path}|${task_id}" \
            >> "${RESULT_DIR}/.gpu_${g}_tasks.txt"
        task_id=$((task_id + 1))
    done
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

        while IFS='|' read -r model dataset gpu2 desc log_path tid; do

            while [ $running -ge $TASKS_PER_GPU ]; do
                wait -n 2>/dev/null
                running=$((running - 1))
                gpu_completed=$((gpu_completed + 1))
            done

            echo "[$(date '+%H:%M:%S')] START  [#${tid}]  model=${model}  dataset=${dataset}  ->  ${gpu}"

            (
                python -u main.py \
                    --model "${model}" \
                    --dataset "${dataset}" \
                    --device "${gpu}" \
                    --description "${desc}" \
                    > "${log_path}" 2>&1
                exit_code=$?
                echo "[$(date '+%H:%M:%S')] DONE   [#${tid}]  model=${model}  dataset=${dataset}  (exit=${exit_code})"
            ) &

            running=$((running + 1))

        done < "${task_file}"

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
printf "%-10s %-10s %-10s %-10s\n" "Model" "Dataset" "HR@20" "NDCG@20"
echo "----------------------------------------"

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        log="${RESULT_DIR}/${model}_${dataset}.log"
        test_line=$(grep "^Test:" "${log}" 2>/dev/null | tail -1)

        if [ -n "${test_line}" ]; then
            hr20=$(echo "${test_line}" | grep -oP "'HR@20':\s*\K[0-9.]+")
            ndcg20=$(echo "${test_line}" | grep -oP "'NDCG@20':\s*\K[0-9.]+")
            [ -z "${hr20}" ] && hr20="N/A"
            [ -z "${ndcg20}" ] && ndcg20="N/A"
            printf "%-10s %-10s %-10s %-10s\n" "${model}" "${dataset}" "${hr20}" "${ndcg20}"
            echo "${hr20} ${ndcg20} ${model} ${dataset}" >> "${RESULT_DIR}/results.txt"
        else
            printf "%-10s %-10s %-10s %-10s\n" "${model}" "${dataset}" "FAIL" "FAIL"
        fi
    done
    echo ""
done

echo ""
echo "============================================"
echo "  Done!"
echo "============================================"
