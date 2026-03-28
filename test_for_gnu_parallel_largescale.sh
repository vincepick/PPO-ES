#!/bin/bash

RESULTS_DIR="output_data/results"

HOSTNAME=$(hostname)
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")

echo "Script started at: $START_TIME on machine: $HOSTNAME"

STEP_SIZES=(256 512 1024 2048 4800)
# All instance 1
# this is controlling the dimensions
# Don't need to do 40 again because thats already done
INSTANCES=(80 160 320 640)

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
    local INSTANCE="$4"

    local EXP_NAME="spaceop${USE_SPACE}_order${INSTANCE_ORDERING}_steps${NUM_STEPS}"
    local EXP_DIR="${RESULTS_DIR}/${EXP_NAME}"
    local OUTPUT_FILE="${EXP_DIR}/${EXP_NAME}_instance${INSTANCE}_output.txt"

    mkdir -p "$EXP_DIR"

    echo "=============================================="
    echo "Running: $EXP_NAME on instance $INSTANCE"
    echo "=============================================="

    python run.py \
        --test_models \
        --type bbob-largescale \
        --instance 1 \
        --dim $INSTANCE \
        --experiment_name "$EXP_NAME" \
        --use_space "$USE_SPACE" \
        --instance_ordering "$INSTANCE_ORDERING" \
        --num_training_instances 12 \
        --num_steps_per_rollout "$NUM_STEPS" \
        > "$OUTPUT_FILE" 2>&1

    echo "Completed: $EXP_NAME on instance $INSTANCE"
}

export -f run_experiment
export RESULTS_DIR

for NUM_STEPS in "${STEP_SIZES[@]}"; do
    for combo in "${COMBINATIONS[@]}"; do
        read -r USE_SPACE INSTANCE_ORDERING <<< "$combo"
        for INSTANCE in "${INSTANCES[@]}"; do
            echo "$NUM_STEPS $USE_SPACE $INSTANCE_ORDERING $INSTANCE"
        done
    done
done | parallel --colsep ' ' -j 40 --joblog parallel_joblog.txt run_experiment {1} {2} {3} {4}

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
DURATION=$(( $(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s) ))

echo "Finished at: $END_TIME"
echo "Total runtime: ${DURATION} seconds"
echo "Results stored in: $RESULTS_DIR"