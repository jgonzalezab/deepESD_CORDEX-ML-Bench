"""Evaluation script for DeepESD model."""

import os
import sys
import torch
import xarray as xr

sys.path.append("/gpfs/projects/meteo/WORK/gonzabad/deep4downscaling")
sys.path.append("/gpfs/projects/meteo/WORK/gonzabad/deepESD_CORDEX-ML-Bench/src")

import deep4downscaling.trans
import deep4downscaling.deep.pred

import models
from config import (
    MODEL_PATH, SUBMISSION_PATH, EVALUATION_EXPERIMENT_SETTINGS,
    PERIOD_DATES, GCM_TRAIN, GCM_EVAL, DATA_PATH
)
from data_utils import (
    load_predictor_and_predictand, load_orography, preprocess_data,
    split_train_test, get_spatial_dims
)


def load_evaluation_predictor(domain: str, period: str, mode: str, 
                             same_gcm_as_train: bool):
    """Load predictor data for evaluation."""
    gcm_name = GCM_TRAIN[domain] if same_gcm_as_train else GCM_EVAL[domain]
    
    if period == 'mid_end_century':
        mid_file = f'{DATA_PATH}/{domain}/test/mid_century/predictors/{mode}/{gcm_name}_2041-2060.nc'
        end_file = f'{DATA_PATH}/{domain}/test/end_century/predictors/{mode}/{gcm_name}_2080-2099.nc'
        
        predictor_mid = xr.open_dataset(mid_file)
        predictor_end = xr.open_dataset(end_file)
        return xr.merge([predictor_mid, predictor_end])
    else:
        period_date = PERIOD_DATES[period]
        filename = f'{DATA_PATH}/{domain}/test/{period}/predictors/{mode}/{gcm_name}_{period_date}.nc'
        return xr.open_dataset(filename)


def main(var_target: str, use_orography: str, domain: str, 
         training_experiment: str, evaluation_experiment: str):
    """Evaluate the DeepESD model."""
    
    # Convert orography flag to boolean
    bool_map = {"True": True, "False": False, "true": True, "false": False}
    use_orog = bool_map.get(use_orography)
    
    # Load training data (for model initialization and statistics)
    print(f"Loading training data for {domain} domain, {var_target} variable...")
    predictor, predictand = load_predictor_and_predictand(
        DATA_PATH, domain, training_experiment, var_target
    )
    orog_data = load_orography(DATA_PATH, domain, training_experiment, use_orog)
    
    # Preprocess
    predictor, predictand, orog_data_stand = preprocess_data(
        predictor, predictand, domain, orog_data
    )
    
    # Split and prepare training data
    x_train, y_train, _, _ = split_train_test(predictor, predictand, training_experiment)
    x_train_standardized = deep4downscaling.trans.standardize(data_ref=x_train, data=x_train)
    
    spatial_dims = get_spatial_dims(domain)
    y_train_stacked = y_train.stack(gridpoint=spatial_dims)
    orog_data_stacked = orog_data_stand.stack(gridpoint=spatial_dims) if orog_data_stand is not None else None
    
    x_train_arr = deep4downscaling.trans.xarray_to_numpy(x_train_standardized)
    y_train_arr = deep4downscaling.trans.xarray_to_numpy(y_train_stacked)
    orog_data_arr = deep4downscaling.trans.xarray_to_numpy(orog_data_stacked) if orog_data_stacked is not None else None
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model name (preserving original format)
    model_name = f'DeepESD_orog{use_orography}_{domain}_{training_experiment}_{var_target}'
    
    model = models.DeepESD(
        x_shape=x_train_arr.shape,
        y_shape=y_train_arr.shape,
        filters_last_conv=1,
        device=device,
        orog_data=orog_data_arr
    )
    model.load_state_dict(torch.load(f'{MODEL_PATH}/{model_name}.pt'))
    print(f"Loaded model: {model_name}")
    
    # Get evaluation settings
    period, mode, same_gcm_as_train = EVALUATION_EXPERIMENT_SETTINGS[training_experiment][evaluation_experiment]
    
    # Load evaluation predictor
    print(f"Loading evaluation data for {evaluation_experiment}...")
    predictor_evaluation = load_evaluation_predictor(domain, period, mode, same_gcm_as_train)
    predictor_evaluation_stand = deep4downscaling.trans.standardize(
        data_ref=x_train, data=predictor_evaluation
    )
    
    # Create mask for mapping predictions
    y_mask = xr.ones_like(y_train.isel(time=0))
    
    # Compute predictions
    print("Computing predictions...")
    pred_evaluation = deep4downscaling.deep.pred.compute_preds_standard(
        x_data=predictor_evaluation_stand,
        model=model,
        device=device,
        var_target=var_target,
        mask=y_mask,
        batch_size=16,
        spatial_dims=spatial_dims
    )
    
    # Save predictions
    orog_label = "OROG" if use_orog else "NO_OROG"
    training_label = training_experiment.replace(
        "ESD_pseudo_reality", "ESD_Pseudo-Reality"
    ).replace("Emulator_hist_future", "Emulator_Hist_Future")
    
    domain_dir = f"{SUBMISSION_PATH}/{domain}_Domain/{training_label}_{orog_label}/{var_target}"
    os.makedirs(domain_dir, exist_ok=True)
    
    prediction_filename = f"{domain_dir}/predictions_{evaluation_experiment}.nc"
    pred_evaluation.to_netcdf(prediction_filename)
    print(f"Predictions saved to: {prediction_filename}")


if __name__ == "__main__":
    var_target = sys.argv[1]
    use_orography = sys.argv[2]
    domain = sys.argv[3]
    training_experiment = sys.argv[4]
    evaluation_experiment = sys.argv[5]
    
    main(var_target, use_orography, domain, training_experiment, evaluation_experiment)
