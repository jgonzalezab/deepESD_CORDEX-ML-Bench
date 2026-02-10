"""Daily field comparison for validation set."""

import os
import sys
sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD_CORDEX-ML-Bench/src')

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from config import VALIDATION_PATH, FIGS_PATH

# Get configuration from environment
var_target = os.getenv('VAR_TARGET', 'pr')
training_experiment = os.getenv('TRAINING_EXPERIMENT', 'ESD_pseudo_reality')
domain = os.getenv('DOMAIN', 'ALPS')
use_orography = os.getenv('USE_OROGRAPHY', 'false').lower() in ('true', '1', 'yes', 'on')

# Selected days for visual comparison
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


def get_lon_lat_dims(data: xr.DataArray) -> dict:
    """Return kwargs so xarray knows which coords represent lon/lat."""
    lon_coord = "lon" if "lon" in data.coords else ("x" if "x" in data.coords else None)
    lat_coord = "lat" if "lat" in data.coords else ("y" if "y" in data.coords else None)

    if lon_coord and lat_coord:
        return {"x": lon_coord, "y": lat_coord}

    dim_names = list(data.dims)
    if len(dim_names) >= 2:
        return {"x": dim_names[-1], "y": dim_names[-2]}
    return {}


# Generate PDF report
os.makedirs(FIGS_PATH, exist_ok=True)
output_pdf = os.path.join(FIGS_PATH, f'{model_name}_daily_comparison.pdf')
print(f"Generating PDF report: {output_pdf}")

with PdfPages(output_pdf) as pdf:
    for day_str in selected_days:
        try:
            # Select the day
            day_gt = groundtruth[var_target].sel(time=day_str, method='nearest')
            day_pred = predictions[var_target].sel(time=day_str, method='nearest')
            
            actual_date = pd.to_datetime(day_gt.time.values).strftime('%Y-%m-%d')
            
            # Compute bias for the day
            day_bias = day_pred - day_gt
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': ccrs.PlateCarree()})
            
            lon_lat_kwargs = get_lon_lat_dims(day_gt)
            
            # Colormaps
            cmap_data = 'turbo' if var_target == 'pr' else 'magma'
            cmap_bias = 'BrBG' if var_target == 'pr' else 'RdBu_r'
            
            # 1. Ground Truth
            vmax = float(day_gt.max())
            vmin = float(day_gt.min())
            
            im0 = day_gt.plot(ax=axes[0], transform=ccrs.PlateCarree(), cmap=cmap_data, 
                             vmin=vmin, vmax=vmax, add_colorbar=False, **lon_lat_kwargs)
            axes[0].set_title(f'Ground Truth ({actual_date})', fontsize=12, fontweight='bold')
            
            # 2. Prediction
            im1 = day_pred.plot(ax=axes[1], transform=ccrs.PlateCarree(), cmap=cmap_data, 
                               vmin=vmin, vmax=vmax, add_colorbar=False, **lon_lat_kwargs)
            axes[1].set_title('Prediction (DeepESD)', fontsize=12, fontweight='bold')
            
            # Add common colorbar for GT and Prediction
            cbar_data = plt.colorbar(im1, ax=axes[:2], orientation='horizontal', pad=0.1, fraction=0.05)
            cbar_data.set_label(f'{var_target} value')
            
            # 3. Bias
            bias_limit = float(np.nanmax(np.abs(day_bias.values)))
            im2 = day_bias.plot(ax=axes[2], transform=ccrs.PlateCarree(), cmap=cmap_bias,
                               vmin=-bias_limit, vmax=bias_limit, add_colorbar=False, **lon_lat_kwargs)
            axes[2].set_title('Bias (Pred - GT)', fontsize=12, fontweight='bold')
            
            cbar_bias = plt.colorbar(im2, ax=axes[2], orientation='horizontal', pad=0.1, fraction=0.05)
            cbar_bias.set_label('Bias')
            
            # Formatting
            for ax in axes:
                ax.coastlines(resolution='50m', linewidth=0.6)
                ax.add_feature(cfeature.BORDERS, linewidth=0.4)
            
            plt.suptitle(f'Daily Comparison | {domain} | {var_target} | {actual_date}', fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Processed day: {actual_date}")
            
        except Exception as e:
            print(f"Error processing day {day_str}: {e}")

print(f"Report saved to: {output_pdf}")
