#!/bin/bash

RESULTS_DIR="output_data/results"

HOSTNAME=$(hostname)
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")

echo "Script started at: $START_TIME on machine: $HOSTNAME"

STEP_SIZES=(256 512 1024 2048 4800)

# Combinations corresponding to the operation of SPACE, and the instance ordering
COMBINATIONS=(
    "0 3"
    "1 3"
    "2 0"
    "2 1"
    "2 2"
    "3 0"
    "3 1"
    "3 2"
)

# Used for preventing the program from trying to multithread, couldn't get it to work properly across multiple cores (40 used) without this 
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TORCH_NUM_THREADS=1


run_experiment() {
    local NUM_STEPS="$1"
    local USE_SPACE="$2"
    local INSTANCE_ORDERING="$3"

    local EXP_NAME="spaceop${USE_SPACE}_order${INSTANCE_ORDERING}_steps${NUM_STEPS}"
    local EXP_DIR="${RESULTS_DIR}/${EXP_NAME}"
    local OUTPUT_FILE="${EXP_DIR}/${EXP_NAME}_output.txt"

    echo "=============================================="
    echo "Running: $EXP_NAME"
    echo "=============================================="

    rm -rf "$EXP_DIR"
    mkdir -p "$EXP_DIR"
    

    python run.py \
        --train \
        --test_models \
        --type bbob \
        --instance 1 \
        --dim 40 \
        --experiment_name "$EXP_NAME" \
        --use_space "$USE_SPACE" \
        --instance_ordering "$INSTANCE_ORDERING" \
        --num_training_instances 12 \
        --num_steps_per_rollout "$NUM_STEPS" \
        > "$OUTPUT_FILE" 2>&1

    echo "Completed: $EXP_NAME"
}

export -f run_experiment
export RESULTS_DIR

for NUM_STEPS in "${STEP_SIZES[@]}"; do
    for combo in "${COMBINATIONS[@]}"; do
        read -r USE_SPACE INSTANCE_ORDERING <<< "$combo"
        echo "$NUM_STEPS $USE_SPACE $INSTANCE_ORDERING"
    done
done | parallel --colsep ' ' -j 40 --joblog parallel_joblog.txt run_experiment {1} {2} {3}

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$(( $(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s) ))

echo "Finished at: $END_TIME"
echo "Total runtime: ${DURATION} seconds"
echo "Results stored in: $RESULTS_DIR"