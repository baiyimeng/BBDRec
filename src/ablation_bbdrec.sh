#!/bin/bash

# ==================== CONFIGURATION ====================
# Ablation study for BBDRec: test the contribution of each component
#
# Variants:
#   baseline    - bbdrec with best hyperparams (pretrain + warmup + attn + MSE)
#   no_pretrain - bbdrec-0 (no pretrained embeddings)
#   no_warmup   - bbdrec-1 (pretrain but no warmup freeze)
#   no_mse      - bbdrec with loss_scale=0 (remove diffusion loss)
#   mlp_decoder - bbdrec with diff_decoder=mlp
#   pcgrad      - bbdrec with PCGrad multi-task optimization
# ====================================================

NUM_GPUS=2
TASKS_PER_GPU=1
GPU_START=0

# Best hyperparameters per dataset (from tuning)
# Format: dataset ds vm ls
declare -A BEST_PARAMS
BEST_PARAMS["baby"]="2 4 1"
BEST_PARAMS["beauty"]="8 0.5 1"
BEST_PARAMS["ml-100k"]="4 0.5 10"
BEST_PARAMS["sports"]="2 0.5 0.01"
BEST_PARAMS["toys"]="16 4 1"
BEST_PARAMS["yelp"]="2 4 0.1"

DATASETS=("baby" "beauty" "ml-100k" "sports" "toys" "yelp")

# Ablation experiments: name|model|extra_args
# extra_args will be appended to the base command
ABLATIONS=(
    "no_pretrain|bbdrec-0|"
    "no_warmup|bbdrec-1|"
    "no_mse|bbdrec|--loss_scale 0"
    "mlp_decoder|bbdrec|--diff_decoder mlp"
)

# ==================== SETUP ====================
RESULT_DIR="logs"
mkdir -p "$RESULT_DIR"

exec > "${RESULT_DIR}/ablation_bbdrec.log" 2>&1

TOTAL=$(( ${#DATASETS[@]} * ${#ABLATIONS[@]} ))
MAX_CONCURRENT=$(( NUM_GPUS * TASKS_PER_GPU ))

echo "============================================"
echo "  BBDRec Ablation Study"
echo "  Datasets: ${DATASETS[*]}"
echo "  Variants: ${#ABLATIONS[@]}"
echo "  Total experiments: ${TOTAL}"
echo "  GPUs: ${NUM_GPUS} (cuda:${GPU_START} ~ cuda:$((GPU_START+NUM_GPUS-1)))"
echo "  Tasks/GPU: ${TASKS_PER_GPU}  |  Max concurrent: ${MAX_CONCURRENT}"
echo "============================================"

# ---- 1. Build per-GPU task files (round-robin assignment) ----
task_id=0
for ((g=0; g<NUM_GPUS; g++)); do
    > "${RESULT_DIR}/.gpu_${g}_tasks.txt"
done

for dataset in "${DATASETS[@]}"; do
    read -r best_ds best_vm best_ls <<< "${BEST_PARAMS[$dataset]}"
    for ablation in "${ABLATIONS[@]}"; do
        IFS='|' read -r abl_name abl_model abl_extra <<< "${ablation}"
        
        g=$(( task_id % NUM_GPUS ))
        gpu="cuda:$((GPU_START + g))"
        desc="${dataset}_${abl_name}"
        log_path="${RESULT_DIR}/${desc}.log"
        
        # Format: dataset|ds|vm|ls|model|extra_args|desc|log_path|task_id|gpu
        echo "${dataset}|${best_ds}|${best_vm}|${best_ls}|${abl_model}|${abl_extra}|${desc}|${log_path}|${task_id}|${gpu}" \
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

        while IFS='|' read -r dataset ds vm ls model extra_args desc log_path tid gpu2; do

            # throttle: wait until this GPU has a free slot
            while [ $running -ge $TASKS_PER_GPU ]; do
                wait -n 2>/dev/null
                running=$((running - 1))
                gpu_completed=$((gpu_completed + 1))
            done

            echo "[$(date '+%H:%M:%S')] START  [#${tid}]  ${desc}  ->  ${gpu}"

            (
                python -u main.py \
                    --dataset "${dataset}" \
                    --model "${model}" \
                    --diffusion_steps "${ds}" \
                    --var_max "${vm}" \
                    --loss_scale "${ls}" \
                    --device "${gpu}" \
                    --description "${desc}" \
                    ${extra_args} \
                    > "${log_path}" 2>&1
                exit_code=$?
                echo "[$(date '+%H:%M:%S')] DONE   [#${tid}]  ${desc}  (exit=${exit_code})"
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
echo "All ${TOTAL} ablation experiments finished!"

# ---- 3. Extract results per dataset ----
echo ""
echo "============================================"
echo "  Ablation Results Summary"
echo "============================================"

for dataset in "${DATASETS[@]}"; do
    read -r best_ds best_vm best_ls <<< "${BEST_PARAMS[$dataset]}"
    
    echo ""
    echo "----------------------------------------"
    echo "  Dataset: ${dataset}  (Best: ds=${best_ds}, vm=${best_vm}, ls=${best_ls})"
    echo "----------------------------------------"
    printf "  %-18s  %-10s  %-10s  %-10s\n" "Variant" "HR@20" "NDCG@20" "ΔHR%"
    echo "  ------------------------------------------------------"
    
    baseline_hr=""
    
    for ablation in "${ABLATIONS[@]}"; do
        IFS='|' read -r abl_name abl_model abl_extra <<< "${ablation}"
        desc="${dataset}_${abl_name}"
        log="${RESULT_DIR}/${desc}.log"
        
        test_line=$(grep "^Test:" "${log}" 2>/dev/null | tail -1)
        
        if [ -n "${test_line}" ]; then
            hr20=$(echo "${test_line}" | grep -oP "'HR@20':\s*\K[0-9.]+")
            ndcg20=$(echo "${test_line}" | grep -oP "'NDCG@20':\s*\K[0-9.]+")
            [ -z "${hr20}" ] && hr20="N/A"
            [ -z "${ndcg20}" ] && ndcg20="N/A"
            
            if [ "${abl_name}" = "baseline" ] && [ "${hr20}" != "N/A" ]; then
                baseline_hr="${hr20}"
            fi
            
            if [ "${hr20}" != "N/A" ] && [ "${baseline_hr}" != "" ] && [ "${abl_name}" != "baseline" ]; then
                delta=$(python3 -c "print(f'{((float(${hr20}) - float(${baseline_hr})) / float(${baseline_hr}) * 100):+.1f}%')" 2>/dev/null)
                [ -z "${delta}" ] && delta="N/A"
                printf "  %-18s  %-10s  %-10s  %-10s\n" "${abl_name}" "${hr20}" "${ndcg20}" "${delta}"
            else
                printf "  %-18s  %-10s  %-10s  %-10s\n" "${abl_name}" "${hr20}" "${ndcg20}" "-"
            fi
        else
            printf "  %-18s  %-10s  %-10s  %-10s\n" "${abl_name}" "FAIL" "FAIL" "-"
        fi
    done
done

echo ""
echo "============================================"
echo "  Ablation Study Complete!"
echo "============================================"
