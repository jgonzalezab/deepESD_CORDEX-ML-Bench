import os
import xarray as xr
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import sys; sys.path.append("/gpfs/projects/meteo/WORK/gonzabad/deep4downscaling")
import deep4downscaling.trans
import deep4downscaling.deep.loss
import deep4downscaling.deep.utils
import deep4downscaling.deep.models
import deep4downscaling.deep.train
import deep4downscaling.deep.pred

# Define PATHs
DATA_PATH = "/gpfs/projects/meteo/WORK/gonzabad/deepESD_CORDEX-ML-Bench/data"
PREDS_PATH = f"{DATA_PATH}/evaluation_preds"
MODEL_PATH = "/gpfs/projects/meteo/WORK/gonzabad/deepESD_CORDEX-ML-Bench/models"

#######################################################################################################
# Define evaluation experiment settings
evaluation_experiment_settings = {
    "ESD_pseudo_reality": {
        "PP_cross_validation": ("historical", "perfect", True),
        "Imperfect_cross_validation": ("historical", "imperfect", True),
        "Extrapolation_perfect": ("mid_end_century", "perfect", True),
        "Extrapolation_imperfect": ("mid_end_century", "imperfect", True),
    },
    "Emulator_hist_future": {
        "PP_cross_validation": ("historical", "perfect", True),
        "Imperfect_cross_validation": ("historical", "imperfect", True),
        "Extrapolation_perfect": ("mid_end_century", "perfect", True),
        "Extrapolation_perfect_hard": ("mid_end_century", "perfect", False),
        "Extrapolation_imperfect_hard": ("mid_end_century", "imperfect", False),
    },
}

# Map periods to dates
period_dates = {
    "historical": "1981-2000",
    "mid_century": "2041-2060",
    "end_century": "2080-2099",
    "mid_end_century": ["2041-2060", "2080-2099"],
}

# GCM selection by domain and training setup
gcm_train = {"NZ": "ACCESS-CM2", "ALPS": "CNRM-CM5"}
gcm_eval = {"NZ": "EC-Earth3", "ALPS": "MPI-ESM-LR"}
#######################################################################################################

#######################################################################################################
# Set the current combination
var_target = 'tasmax'
domain = 'ALPS'
training_experiment = 'Emulator_hist_future'
evaluation_experiment = 'Extrapolation_imperfect_hard'
#######################################################################################################

#######################################################################################################
"""
Despite this being an evaluation script we still need to load the predictor and predictand
to properly initialize the DL architecture and the statistics for the standardization.
"""

# Get settings for the current combination
if training_experiment == 'ESD_pseudo_reality':
    period_training = '1961-1980'
elif training_experiment == 'Emulator_hist_future':
    period_training = '1961-1980_2080-2099'
else:
    raise ValueError('Provide a valid date')

# Set the GCM
if domain == 'ALPS':
    gcm_name = 'CNRM-CM5'
elif domain == 'NZ':
    gcm_name = 'ACCESS-CM2'

# Load predictor
predictor_filename = f'{DATA_PATH}/{domain}_domain/train/{training_experiment}/predictors/{gcm_name}_{period_training}.nc'
predictor = xr.open_dataset(predictor_filename)

# Load predictand
predictand_filename = f'{DATA_PATH}/{domain}_domain/train/{training_experiment}/target/pr_tasmax_{gcm_name}_{period_training}.nc'
predictand = xr.open_dataset(predictand_filename)
predictand = predictand[[var_target]]

# Remove days with nans in the predictor
predictor = deep4downscaling.trans.remove_days_with_nans(predictor)

# Align both datasets in time
predictor, predictand = deep4downscaling.trans.align_datasets(predictor, predictand, 'time')

# Set a test set for some evaluation
if training_experiment == 'ESD_pseudo_reality':
    years_train = list(range(1961, 1975))
    years_test = list(range(1975, 1980+1))
elif training_experiment == 'Emulator_hist_future':
    years_train = list(range(1961, 1980+1)) + list(range(2080, 2090))
    years_test = list(range(2090, 2099+1))

x_train = predictor.sel(time=np.isin(predictor['time'].dt.year, years_train))
y_train = predictand.sel(time=np.isin(predictand['time'].dt.year, years_train))

x_test = predictor.sel(time=np.isin(predictor['time'].dt.year, years_test))
y_test = predictand.sel(time=np.isin(predictand['time'].dt.year, years_test))

# Standardize the predictor
x_train_stand = deep4downscaling.trans.standardize(data_ref=x_train, data=x_train)

# Flat the predictand
if domain == 'ALPS':
    spatial_dims = ('x', 'y')
elif domain == 'NZ':
    spatial_dims = ('lat', 'lon')

y_train_stack = y_train.stack(gridpoint=spatial_dims)

# Transform to numpy arrays
x_train_stand_arr = deep4downscaling.trans.xarray_to_numpy(x_train_stand)
y_train_arr = deep4downscaling.trans.xarray_to_numpy(y_train_stack)
#######################################################################################################

#######################################################################################################
# Load the model
model_name = f'DeepESD_{domain}_{training_experiment}_{var_target}'
model = deep4downscaling.deep.models.DeepESDtas(x_shape=x_train_stand_arr.shape,
                                                y_shape=y_train_arr.shape,
                                                filters_last_conv=1,
                                                stochastic=False)

# Load the model
model.load_state_dict(torch.load(f'{MODEL_PATH}/{model_name}.pt'))
#######################################################################################################

#######################################################################################################
# Get settings for the chosen experiments
period, mode, same_gcm_as_train = evaluation_experiment_settings[training_experiment][evaluation_experiment]
period_date = period_dates[period]
if same_gcm_as_train:
    gcm_name = gcm_train[domain]
else:
    gcm_name = gcm_eval[domain]


if period == 'mid_end_century':
    predictor_evaluation_filename_mid_century = f'{DATA_PATH}/{domain}_domain/test/mid_century/predictors/{mode}/{gcm_name}_2041-2060.nc'
    predictor_evaluation_mid = xr.open_dataset(predictor_evaluation_filename_mid_century)
    
    predictor_evaluation_filename_end_century = f'{DATA_PATH}/{domain}_domain/test/end_century/predictors/{mode}/{gcm_name}_2080-2099.nc'
    predictor_evaluation_end = xr.open_dataset(predictor_evaluation_filename_end_century)

    predictor_evaluation = xr.merge([predictor_evaluation_mid, predictor_evaluation_end])
else:
    predictor_evaluation_filename = f'{DATA_PATH}/{domain}_domain/test/{period}/predictors/{mode}/{gcm_name}_{period_date}.nc'
    predictor_evaluation = xr.open_dataset(predictor_evaluation_filename)

# Standardize the predictor
predictor_evaluation_stand = deep4downscaling.trans.standardize(data_ref=x_train, data=predictor_evaluation)

# Compute map for mapping from Torch to NetCDF
y_mask = xr.ones_like(y_train.isel(time=0))

# Set the device
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Compute predictions
if domain == 'ALPS':
    spatial_dims = ('x', 'y')
elif domain == 'NZ':
    spatial_dims = ('lat', 'lon')

pred_evaluation = deep4downscaling.deep.pred.compute_preds_standard(
                                    x_data=predictor_evaluation_stand, model=model,
                                    device=device, var_target=var_target,
                                    mask=y_mask, batch_size=16,
                                    spatial_dims=spatial_dims)

# Save predictions
pred_evaluation.to_netcdf(f'{PREDS_PATH}/{model_name}_{evaluation_experiment}.nc')
#######################################################################################################