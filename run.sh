#!/bin/bash
#SBATCH --job-name=deepesd_train_eval
#SBATCH --output=logs/train_eval_%j.out
#SBATCH --error=logs/train_eval_%j.out
#SBATCH --partition=wngpu
#SBATCH --mem-per-cpu=127000
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
orography_values=("True" "False")
domains=("ALPS" "NZ" "SA")
training_experiments=("ESD_pseudo_reality" "Emulator_hist_future")

# Evaluation experiments for each training experiment
declare -A eval_experiments
eval_experiments["ESD_pseudo_reality"]="PP_cross_validation Imperfect_cross_validation Extrapolation_perfect Extrapolation_imperfect"
eval_experiments["Emulator_hist_future"]="PP_cross_validation Imperfect_cross_validation Extrapolation_perfect Extrapolation_perfect_hard Extrapolation_imperfect_hard"

# Counter for total combinations
total_combinations=0
completed_combinations=0
failed_combinations=0

# Calculate total combinations for progress tracking
for var_target in "${var_targets[@]}"; do
    for orog in "${orography_values[@]}"; do
        for domain in "${domains[@]}"; do
            for training_experiment in "${training_experiments[@]}"; do
                total_combinations=$((total_combinations + 1))
            done
        done
    done
done

echo "=========================================="
echo "Running training and evaluation for all combinations"
echo "Total training combinations: $total_combinations"
echo "=========================================="
echo ""

# Loop over all combinations
for var_target in "${var_targets[@]}"; do
    for orog in "${orography_values[@]}"; do
        for domain in "${domains[@]}"; do
            for training_experiment in "${training_experiments[@]}"; do
                completed_combinations=$((completed_combinations + 1))
                
                echo "=========================================="
                echo "Combination $completed_combinations/$total_combinations"
                echo "var_target: $var_target"
                echo "orography: $orog"
                echo "domain: $domain"
                echo "training_experiment: $training_experiment"
                echo "=========================================="
                echo ""
                
                # Run training
                echo "Starting training..."
                python scripts/train.py "$var_target" "$orog" "$domain" "$training_experiment"
            
                # Check if training was successful
                if [ $? -ne 0 ]; then
                    echo "ERROR: Training failed for var_target=$var_target, orography=$orog, domain=$domain, training_experiment=$training_experiment"
                    failed_combinations=$((failed_combinations + 1))
                    echo "Skipping evaluations for this combination."
                    echo ""
                    continue
                fi
                
                echo "Training completed successfully!"
                echo ""
                
                # Run all evaluations for this training experiment
                eval_list=${eval_experiments[$training_experiment]}
                eval_count=0
                for evaluation_experiment in $eval_list; do
                    eval_count=$((eval_count + 1))
                    echo "  Evaluation $eval_count: $evaluation_experiment"
                    
                    python scripts/evaluation.py "$var_target" "$orog" "$domain" "$training_experiment" "$evaluation_experiment"
                
                    # Check if evaluation was successful
                    if [ $? -ne 0 ]; then
                        echo "  ERROR: Evaluation failed for evaluation_experiment=$evaluation_experiment"
                        failed_combinations=$((failed_combinations + 1))
                    else
                        echo "  Evaluation $evaluation_experiment completed successfully!"
                    fi
                    echo ""
                done
                
                echo "Completed all evaluations for this combination."
                echo ""
            done
        done
    done
done

echo "=========================================="
echo "All combinations processed!"
echo "Total combinations: $total_combinations"
echo "Successfully completed: $((total_combinations - failed_combinations))"
echo "Failed: $failed_combinations"
echo "=========================================="

