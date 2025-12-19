#!/bin/bash

source .venv/bin/activate

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Path to experiment configs
CONFIGS_DIR="$PROJECT_ROOT/annotix_ml/graphtransf/configs/arch"

# Path to training script
TRAIN_SCRIPT="$PROJECT_ROOT/annotix_ml/graphtransf/train/train.py"

# Create logs directory for experiment outputs
LOGS_DIR="$PROJECT_ROOT/logs/arch"
mkdir -p "$LOGS_DIR"

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Starting all experiments..."
echo "Logs will be saved to: $LOGS_DIR"
echo "================================"

# Loop through all YAML config files recursively in the experiments directory
find "$CONFIGS_DIR" -type f -name "*.yaml" | sort | while read -r config_file; do
    folder_name=$(basename "$(dirname "$config_file")")
    file_name=$(basename "$config_file" .yaml)
    config_name="${folder_name}_${file_name}"

    # Define extra_features combinations
    features_combinations=(
        "laplacian_embedding"
        "node_cycle"
        "valence_features"
        "laplacian_embedding,node_cycle"
        "laplacian_embedding,valence_features"
        "node_cycle,valence_features"
        "laplacian_embedding,node_cycle,valence_features"
    )

    for features in "${features_combinations[@]}"; do
        # Create a tag for the features to be used in project name and paths
        # Replace commas with underscores and remove spaces if any
        feature_tag=$(echo "$features" | sed 's/,/_/g' | tr -d ' ')

        echo "Running experiment: $config_name with features: $features"

        # Construct dynamic project name and save path
        project_name="annotix_ml_${config_name}_${feature_tag}"
        save_path="$LOGS_DIR/${config_name}/${feature_tag}"

        python "$TRAIN_SCRIPT" \
            --config "$config_file" \
            --extra-features "$features" \
            --project-name "$project_name" \
            --save-path "$save_path"
    done
done
