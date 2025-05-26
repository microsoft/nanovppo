#!/bin/bash

# Example script to run run_bestofn.py for evaluating a trained model with pass@N metrics
# This script is used to evaluate the quality of a previously trained model by generating
# multiple responses for each prompt and computing pass@N statistics.

MODEL_DIR="./runs_outputs/your_model_path"
OUTPUT_DIR="./runs_outputs/bestofn_results"
SEED=42
MAX_N=32  # Number of responses to generate for each prompt

python run_bestofn.py \
    --load_model $MODEL_DIR \
    -o $OUTPUT_DIR \
    -s $SEED \
    -maxN $MAX_N