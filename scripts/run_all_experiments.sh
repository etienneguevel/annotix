#!/bin/bash

# Script to run all graph transformer experiments
# Usage: ./scripts/run_all_experiments.sh

set -e  # Exit on error

# Activate virtual environment
source .venv/bin/activate

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Path to experiment configs
CONFIGS_DIR="$PROJECT_ROOT/annotix_ml/graphtransf/configs/experiments"

# Path to training script
TRAIN_SCRIPT="$PROJECT_ROOT/annotix_ml/graphtransf/train/train.py"

# Create logs directory for experiment outputs
LOGS_DIR="$PROJECT_ROOT/logs/experiments"
mkdir -p "$LOGS_DIR"

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Starting all experiments..."
echo "Logs will be saved to: $LOGS_DIR"
echo "================================"

# Loop through all YAML config files in the experiments directory
for config_file in "$CONFIGS_DIR"/*.yaml; do
    if [ -f "$config_file" ]; then
        config_name=$(basename "$config_file" .yaml)
        log_file="$LOGS_DIR/${config_name}_${TIMESTAMP}.txt"

        echo ""
        echo "Running experiment: $config_name"
        echo "Log file: $log_file"
        echo "--------------------------------"

        # Run the training script with the current config and save logs
        python "$TRAIN_SCRIPT" --config "$config_file" 2>&1 | tee "$log_file"

        echo "Completed: $config_name"
    fi
done

echo ""
echo "================================"
echo "All experiments completed!"
