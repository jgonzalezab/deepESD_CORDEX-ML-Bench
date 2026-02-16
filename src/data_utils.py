"""Utilities for loading and preprocessing data."""

import xarray as xr
import numpy as np
import sys
sys.path.append("/gpfs/projects/meteo/WORK/gonzabad/deep4downscaling")
import deep4downscaling.trans


def get_training_period(training_experiment: str) -> str:
    """Get the training period for the given experiment."""
    
    period_map = {'ESD_pseudo_reality': '1961-1980',
                  'Emulator_hist_future': '1961-1980_2080-2099'}
    if training_experiment not in period_map:
        raise ValueError(f'Invalid training experiment: {training_experiment}')
    return period_map[training_experiment]


def get_gcm_name(domain: str) -> str:
    """Get the GCM name for the given domain (for training)."""
    gcm_map = {"ALPS": "CNRM-CM5", "NZ": "ACCESS-CM2", "SA": "ACCESS-CM2"}
    if domain not in gcm_map:
        raise ValueError(f'Invalid domain: {domain}')
    return gcm_map[domain]


def get_spatial_dims(domain: str) -> tuple:
    """Get the spatial dimensions for the given domain."""
    dims_map = {"ALPS": ('y', 'x'), "NZ": ('lat', 'lon'), "SA": ('lat', 'lon')}
    if domain not in dims_map:
        raise ValueError(f'Invalid domain: {domain}')
    return dims_map[domain]


def load_predictor_and_predictand(data_path: str, domain: str, 
                                   training_experiment: str, var_target: str):
    """Load predictor and predictand datasets."""
    period_training = get_training_period(training_experiment)
    gcm_name = get_gcm_name(domain)
    
    predictor_filename = f'{data_path}/{domain}/train/{training_experiment}/predictors/{gcm_name}_{period_training}.nc'
    predictor = xr.open_dataset(predictor_filename)
    if domain == 'SA': 
        predictor = predictor.drop_vars('time_bnds')
    
    predictand_filename = f'{data_path}/{domain}/train/{training_experiment}/target/pr_tasmax_{gcm_name}_{period_training}.nc'
    predictand = xr.open_dataset(predictand_filename)
    predictand = predictand[[var_target]]
    
    return predictor, predictand


def preprocess_data(predictor: xr.Dataset, predictand: xr.Dataset, 
                    domain: str):
    """Remove NaNs and align datasets."""
    predictor = deep4downscaling.trans.remove_days_with_nans(predictor)
    predictor, predictand = deep4downscaling.trans.align_datasets(predictor, predictand, 'time')
    
    return predictor, predictand


def split_train_test(predictor: xr.Dataset, predictand: xr.Dataset, 
                     training_experiment: str, validation_mode: bool = True):
    """Split data into training and test sets.
    
    Args:
        predictor: Predictor dataset.
        predictand: Predictand dataset.
        training_experiment: Training experiment name.
        validation_mode: If True, reserve years for validation. If False, use all data for training.
    """
    if training_experiment == 'ESD_pseudo_reality':
        if validation_mode:
            years_train = list(range(1961, 1980))
            years_test = list(range(1980, 1981))
        else:
            years_train = list(range(1961, 1981))
            years_test = []
    elif training_experiment == 'Emulator_hist_future':
        if validation_mode:
            years_train = list(range(1961, 1981)) + list(range(2080, 2098))
            years_test = list(range(2098, 2100))
        else:
            years_train = list(range(1961, 1981)) + list(range(2080, 2100))
            years_test = []
    else:
        raise ValueError(f'Invalid training experiment: {training_experiment}')
    
    x_train = predictor.sel(time=np.isin(predictor['time'].dt.year, years_train))
    y_train = predictand.sel(time=np.isin(predictand['time'].dt.year, years_train))
    x_test = predictor.sel(time=np.isin(predictor['time'].dt.year, years_test))
    y_test = predictand.sel(time=np.isin(predictand['time'].dt.year, years_test))
    
    return x_train, y_train, x_test, y_test

def load_orography(data_path: str, domain: str, training_experiment: str):
    """Load the orography data."""
    period_training = get_training_period(training_experiment)
    gcm_name = get_gcm_name(domain)
    orog_path = f'{data_path}/{domain}/train/{training_experiment}/predictors/Static_fields.nc'
    orog = xr.open_dataset(orog_path)
    orog = orog[['orog']]
    return orog