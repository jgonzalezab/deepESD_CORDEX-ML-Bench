"""Compute global Wasserstein distance for validation set (no figures)."""

import os
import sys
sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD_CORDEX-ML-Bench/src')
sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/ml-benchmark')

import xarray as xr
from evaluation import diagnostics as eval_diagnostics

from config import VALIDATION_PATH

# Get configuration from environment
var_target = os.getenv('VAR_TARGET', 'pr')
training_experiment = os.getenv('TRAINING_EXPERIMENT', 'ESD_pseudo_reality')
domain = os.getenv('DOMAIN', 'ALPS')

# Model name
model_name = f'DeepESD_{training_experiment}_{domain}_{var_target}'

# Load predictions and ground truth
predictions_path = os.path.join(VALIDATION_PATH, f'{model_name}_predictions.nc')
groundtruth_path = os.path.join(VALIDATION_PATH, f'{model_name}_groundtruth.nc')

if not os.path.exists(predictions_path) or not os.path.exists(groundtruth_path):
    print(f"Error: Prediction or ground truth files not found.")
    print(f"Expected: {predictions_path}")
    print(f"Expected: {groundtruth_path}")
    print("Run predict_validation.py first.")
    sys.exit(1)

predictions = xr.open_dataset(predictions_path)
groundtruth = xr.open_dataset(groundtruth_path)

# Compute global Wasserstein distance
print("Computing global Wasserstein distance...")
wasserstein_global = eval_diagnostics.wasserstein_distance(
    groundtruth, predictions, var=var_target, spatial=False
)
print(f"Global Wasserstein distance: {wasserstein_global:.4f}")
