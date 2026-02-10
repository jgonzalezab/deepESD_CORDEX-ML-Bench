"""Compute standard metrics for validation set."""

import os
import sys
sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/deepESD_CORDEX-ML-Bench/src')
sys.path.append('/gpfs/projects/meteo/WORK/gonzabad/ml-benchmark')

import xarray as xr
from evaluation import diagnostics as eval_diagnostics
from evaluation import indices as eval_indices
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from config import VALIDATION_PATH, FIGS_PATH

# Get configuration from environment
var_target = os.getenv('VAR_TARGET', 'pr')
training_experiment = os.getenv('TRAINING_EXPERIMENT', 'ESD_pseudo_reality')
domain = os.getenv('DOMAIN', 'ALPS')
use_orography = os.getenv('USE_OROGRAPHY', 'false').lower() in ('true', '1', 'yes', 'on')

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

# Compute diagnostics
rmse = eval_diagnostics.rmse(groundtruth, predictions, var=var_target, dim='time')
bias_mean = eval_diagnostics.bias_index(groundtruth, predictions, eval_indices.mean, var=var_target)
wasserstein = eval_diagnostics.wasserstein_distance(groundtruth, predictions, var=var_target, spatial=True)

if var_target == 'tasmax':
    extra_metrics = {
        'bias_p98': eval_diagnostics.bias_index(groundtruth, predictions, eval_indices.quantile, var=var_target, q=0.98),
        'bias_txx': eval_diagnostics.bias_index(groundtruth, predictions, eval_indices.txx, var=var_target)
    }
elif var_target == 'pr':
    extra_metrics = {
        'bias_sdii': eval_diagnostics.bias_index(groundtruth, predictions, eval_indices.sdii, var=var_target),
        'bias_rx1day': eval_diagnostics.bias_index(groundtruth, predictions, eval_indices.rx1day, var=var_target)
    }
else:
    raise ValueError('Unsupported variable target')

diagnostics = {'rmse': rmse, 'bias_mean': bias_mean, 'wasserstein': wasserstein, **extra_metrics}

# Metric configuration
metric_titles = {
    'rmse': 'RMSE',
    'bias_mean': 'Mean Bias',
    'bias_p98': 'Bias (p98)',
    'bias_txx': 'Bias (TXx)',
    'bias_sdii': 'Bias (SDII)',
    'bias_rx1day': 'Bias (RX1day)',
    'wasserstein': 'Wasserstein Distance'
}

bias_metrics = {'bias_mean', 'bias_p98', 'bias_txx', 'bias_sdii', 'bias_rx1day'}

METRIC_COLORBAR_LIMITS = {
    "tasmax": {
        "rmse": {"vmin": 0.0, "vmax": 4.0},
        "bias_mean": {"vmin": -1.0, "vmax": 1.0},
        "bias_p98": {"vmin": -3.0, "vmax": 3.0},
        "bias_txx": {"vmin": -3.0, "vmax": 3.0},
        "wasserstein": {"vmin": 0.0, "vmax": 2.0},
    },
    "pr": {
        "rmse": {"vmin": 0.0, "vmax": 12.0},
        "bias_mean": {"vmin": -2.0, "vmax": 2.0},
        "bias_sdii": {"vmin": -2.0, "vmax": 2.0},
        "bias_rx1day": {"vmin": -40.0, "vmax": 40.0},
        "wasserstein": {"vmin": 0.0, "vmax": 5.0},
    },
}

METRIC_BOXPLOT_LIMITS = {
    "tasmax": {
        "rmse": {"ymin": 0.0, "ymax": 4.0},
        "bias_mean": {"ymin": -1.0, "ymax": 1.0},
        "bias_p98": {"ymin": -3.0, "ymax": 3.0},
        "bias_txx": {"ymin": -3.0, "ymax": 3.0},
        "wasserstein": {"ymin": 0.0, "ymax": 2.0},
    },
    "pr": {
        "rmse": {"ymin": 0.0, "ymax": 12.0},
        "bias_mean": {"ymin": -2.0, "ymax": 2.0},
        "bias_sdii": {"ymin": -2.0, "ymax": 2.0},
        "bias_rx1day": {"ymin": -40.0, "ymax": 40.0},
        "wasserstein": {"ymin": 0.0, "ymax": 5.0},
    },
}


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
output_pdf = os.path.join(FIGS_PATH, f'{model_name}_metrics_report.pdf')
print(f"Generating PDF report: {output_pdf}")

with PdfPages(output_pdf) as pdf:
    for name, value in diagnostics.items():
        # Extract DataArray from Dataset if needed
        if isinstance(value, xr.Dataset):
            if var_target in value.data_vars:
                value = value[var_target]
            else:
                value = value[list(value.data_vars)[0]]
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(15, 7))
        
        # 1. Spatial Map
        ax_map = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        
        lon_lat_kwargs = get_lon_lat_dims(value)
        
        # Determine colormap
        is_bias = name in bias_metrics
        if is_bias:
            cmap = 'BrBG' if var_target == 'pr' else 'RdBu_r'
        else:
            cmap = 'viridis'
        
        # Get colorbar limits from configuration
        var_config = METRIC_COLORBAR_LIMITS.get(var_target, {})
        metric_config = var_config.get(name)
        
        if metric_config is not None:
            vmin = metric_config.get("vmin")
            vmax = metric_config.get("vmax")
            if is_bias and vmin is not None and vmax is not None:
                pass
            elif is_bias and (vmin is None or vmax is None):
                v_limit = float(np.nanmax(np.abs(value.values)))
                vmin, vmax = -v_limit, v_limit
        else:
            if is_bias:
                v_limit = float(np.nanmax(np.abs(value.values)))
                vmin, vmax = -v_limit, v_limit
            else:
                vmin, vmax = None, None
        
        # Plot spatial data
        plot_kwargs = {
            'ax': ax_map,
            'transform': ccrs.PlateCarree(),
            'cmap': cmap,
            'add_colorbar': False,
        }
        if vmin is not None and vmax is not None:
            plot_kwargs['vmin'] = vmin
            plot_kwargs['vmax'] = vmax
        
        plot_kwargs.update(lon_lat_kwargs)
        
        value.plot(**plot_kwargs)
        ax_map.set_title("")
        ax_map.coastlines(resolution='50m', linewidth=0.6)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax_map.set_title('Spatial Map', fontsize=12, fontweight='bold')
        
        # Add colorbar
        vmin_cbar = vmin if vmin is not None else float(value.min().values)
        vmax_cbar = vmax if vmax is not None else float(value.max().values)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin_cbar, vmax=vmax_cbar))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_map, orientation='horizontal', pad=0.1, fraction=0.05)
        cbar.set_label(metric_titles.get(name, name))
        
        # 2. Boxplot
        ax_box = fig.add_subplot(1, 2, 2)
        flat_data = value.values.flatten()
        flat_data = flat_data[np.isfinite(flat_data)]
        
        bp = ax_box.boxplot(flat_data, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax_box.set_title('Distribution', fontsize=12, fontweight='bold')
        ax_box.set_ylabel(metric_titles.get(name, name))
        ax_box.set_axisbelow(True)
        
        # Get y-axis limits from configuration
        var_boxplot_config = METRIC_BOXPLOT_LIMITS.get(var_target, {})
        boxplot_config = var_boxplot_config.get(name)
        
        if boxplot_config is not None:
            ymin = boxplot_config.get("ymin")
            ymax = boxplot_config.get("ymax")
            if ymin is not None and ymax is not None:
                ax_box.set_ylim(ymin, ymax)
            elif ymin is not None:
                ax_box.set_ylim(bottom=ymin)
            elif ymax is not None:
                ax_box.set_ylim(top=ymax)
        
        # Add mean value
        mean_val = np.nanmean(flat_data)
        ax_box.text(0.95, 0.95, f'Mean: {mean_val:.2f}',
                    transform=ax_box.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10, fontweight='bold')
        
        # Overall title
        plt.suptitle(f'{metric_titles.get(name, name)} | DeepESD | {var_target}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        pdf.savefig(fig)
        plt.close(fig)

print(f"Report saved to: {output_pdf}")
