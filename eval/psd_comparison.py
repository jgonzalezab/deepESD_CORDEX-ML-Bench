"""Compute power spectral density comparison for validation set."""

import os
import sys
sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD_CORDEX-ML-Bench/src')
sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/ml-benchmark')

import xarray as xr
import matplotlib.pyplot as plt
from evaluation import diagnostics as eval_diagnostics
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from config import VALIDATION_PATH, FIGS_PATH

# Get configuration from environment
var_target = os.getenv('VAR_TARGET', 'pr')
training_experiment = os.getenv('TRAINING_EXPERIMENT', 'ESD_pseudo_reality')
domain = os.getenv('DOMAIN', 'ALPS')
use_orography = os.getenv('USE_OROGRAPHY', 'false').lower() in ('true', '1', 'yes', 'on')

# Selected days for daily PSD analysis
selected_days = ['1980-01-15', '1980-07-15'] if training_experiment == 'ESD_pseudo_reality' else ['2098-01-15', '2098-07-15']

# Model name
orog_suffix = '-orog' if use_orography else ''
model_name = f'DeepESD_{training_experiment}_{domain}_{var_target}{orog_suffix}'

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

# Generate PDF report
os.makedirs(FIGS_PATH, exist_ok=True)
output_pdf = os.path.join(FIGS_PATH, f'{model_name}_psd_comparison.pdf')
print(f"Generating PDF report: {output_pdf}")

with PdfPages(output_pdf) as pdf:
    # 1. Overall PSD for the whole validation period
    print("Computing overall PSD...")
    psd_gt_all, psd_pred_all = eval_diagnostics.psd(x0=groundtruth, x1=predictions, var=var_target)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.loglog(psd_gt_all.wavenumber, psd_gt_all, label="Ground Truth", color='black', linewidth=2)
    ax.loglog(psd_pred_all.wavenumber, psd_pred_all, label="Prediction (DeepESD)", color='red', linestyle='--', linewidth=2)
    
    ax.set_title(f'Overall Power Spectral Density | {domain} | {var_target}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Wavenumber', fontsize=12)
    ax.set_ylabel('Power', fontsize=12)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # 2. Daily PSD for selected days
    for day_str in selected_days:
        try:
            # Select the day keeping the time dimension
            day_gt_ds = groundtruth.sel(time=[day_str], method='nearest')
            day_pred_ds = predictions.sel(time=[day_str], method='nearest')
            
            actual_date = pd.to_datetime(day_gt_ds.time.values[0]).strftime('%Y-%m-%d')
            
            print(f"Computing PSD for {actual_date}...")
            psd_gt, psd_pred = eval_diagnostics.psd(x0=day_gt_ds, x1=day_pred_ds, var=var_target)
            
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.loglog(psd_gt.wavenumber, psd_gt, label="Ground Truth", color='black', linewidth=2)
            ax.loglog(psd_pred.wavenumber, psd_pred, label="Prediction (DeepESD)", color='red', linestyle='--', linewidth=2)
            
            ax.set_title(f'Power Spectral Density | {domain} | {var_target} | {actual_date}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Wavenumber', fontsize=12)
            ax.set_ylabel('Power', fontsize=12)
            ax.grid(True, which="both", ls="-", alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            
        except Exception as e:
            print(f"Error processing day {day_str}: {e}")

print(f"Report saved to: {output_pdf}")
