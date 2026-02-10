#!/bin/bash
#SBATCH --job-name=deepesd_train_eval
#SBATCH --output=logs/train_eval_%j.out
#SBATCH --error=logs/train_eval_%j.out
#SBATCH --partition=wngpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=72:00:00

# Set the working directory
cd /gpfs/projects/meteo/WORK/gonzabad/deepESD_CORDEX-ML-Bench

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source /gpfs/projects/meteo/WORK/gonzabad/miniforge3/etc/profile.d/conda.sh
conda activate pnacc-gpu

# Define all combinations to run
var_targets=("pr" "tasmax")
domains=("ALPS" "NZ" "SA")
training_experiments=("ESD_pseudo_reality" "Emulator_hist_future")

# Counter for total combinations
total_combinations=0
completed_combinations=0
failed_combinations=0

# Calculate total combinations for progress tracking
for var_target in "${var_targets[@]}"; do
    for domain in "${domains[@]}"; do
        for training_experiment in "${training_experiments[@]}"; do
            total_combinations=$((total_combinations + 1))
        done
    done
done

echo "=========================================="
echo "Running training for all combinations"
echo "Total training combinations: $total_combinations"
echo "=========================================="
echo ""

# Loop over all combinations for training
for var_target in "${var_targets[@]}"; do
    for domain in "${domains[@]}"; do
        for training_experiment in "${training_experiments[@]}"; do
            completed_combinations=$((completed_combinations + 1))
            
            echo "=========================================="
            echo "Combination $completed_combinations/$total_combinations"
            echo "var_target: $var_target"
            echo "domain: $domain"
            echo "training_experiment: $training_experiment"
            echo "=========================================="
            echo ""
            
            # Run training
            echo "Starting training..."
            python scripts/train.py "$var_target" "$domain" "$training_experiment"
        
            # Check if training was successful
            if [ $? -ne 0 ]; then
                echo "ERROR: Training failed for var_target=$var_target, domain=$domain, training_experiment=$training_experiment"
                failed_combinations=$((failed_combinations + 1))
            else
                echo "Training completed successfully!"
            fi
            echo ""
        done
    done
done

# echo "=========================================="
# echo "Training finished. Starting submission generation..."
# echo "=========================================="
# echo ""

# # Run submission generation
# python scripts/submission.py

# if [ $? -ne 0 ]; then
#     echo "ERROR: Submission generation failed."
#     exit 1
# fi

# echo "=========================================="
# echo "All steps completed!"
# echo "Total training combinations: $total_combinations"
# echo "Successfully trained: $((total_combinations - failed_combinations))"
# echo "Failed training: $failed_combinations"
# echo "Submission package created successfully."
# echo "=========================================="

