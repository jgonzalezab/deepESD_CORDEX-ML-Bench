"""Compute predictions on validation set for model evaluation."""

import os
import sys
import torch
import xarray as xr

sys.path.append("/gpfs/projects/meteo/WORK/gonzabad/deep4downscaling")
sys.path.append("/gpfs/projects/meteo/WORK/gonzabad/deepESD_CORDEX-ML-Bench/src")

import deep4downscaling.trans
import deep4downscaling.deep.pred
from models import DeepESDpr, DeepESDtas

from config import MODEL_PATH, DATA_PATH, VALIDATION_PATH
from data_utils import (load_predictor_and_predictand, preprocess_data,
                        split_train_test, get_spatial_dims, load_orography)


def main(var_target: str, domain: str, training_experiment: str, use_orography: bool = False):
    """Compute predictions on validation set."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data for {domain} domain, {var_target} variable...")
    predictor, predictand = load_predictor_and_predictand(DATA_PATH, domain, training_experiment, var_target)
    
    # Preprocess
    predictor, predictand = preprocess_data(predictor, predictand, domain)
    
    # Split into train and test (validation mode)
    x_train, y_train, x_test, y_test = split_train_test(predictor, predictand, training_experiment, validation_mode=True)
    
    if len(x_test.time) == 0:
        print("No validation data available for this experiment.")
        return
    
    print(f"Validation set size: {len(x_test.time)} days")
    
    # Standardize test predictor using training statistics
    x_test_stand = deep4downscaling.trans.standardize(data_ref=x_train, data=x_test)
    
    # Get shapes for model initialization
    x_train_standardized = deep4downscaling.trans.standardize(data_ref=x_train, data=x_train)
    x_train_arr = deep4downscaling.trans.xarray_to_numpy(x_train_standardized)
    spatial_dims = get_spatial_dims(domain)
    y_train_stacked = y_train.stack(gridpoint=spatial_dims)
    y_train_arr = deep4downscaling.trans.xarray_to_numpy(y_train_stacked)
    
    # Load orography if needed
    if use_orography:
        orog = load_orography(DATA_PATH, domain, training_experiment)
        orog = orog / orog.max()  # Normalize to 0-1
        orog_arr = torch.tensor(deep4downscaling.trans.xarray_to_numpy(orog),
                               dtype=torch.float32)
    else:
        orog_arr = None
    
    # Initialize and load model
    if var_target == 'pr':
        model = DeepESDpr(x_shape=x_train_arr.shape,
                          y_shape=y_train_arr.shape,
                          filters_last_conv=1,
                          stochastic=False,
                          last_relu=False,
                          orography=orog_arr)
    else:
        model = DeepESDtas(x_shape=x_train_arr.shape,
                           y_shape=y_train_arr.shape,
                           filters_last_conv=1,
                           stochastic=False,
                           orography=orog_arr)
    
    model_suffix = '-orog' if use_orography else ''
    model_name = f'DeepESD_{training_experiment}_{domain}_{var_target}{model_suffix}.pt'
    model_path = os.path.join(MODEL_PATH, model_name)
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first.")
        return
    
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()
    
    # Compute predictions
    print("Computing predictions...")
    y_mask = xr.ones_like(y_train.isel(time=0))
    pred_da = deep4downscaling.deep.pred.compute_preds_standard(x_data=x_test_stand,
                                                                model=model,
                                                                device=device,
                                                                var_target=var_target,
                                                                mask=y_mask,
                                                                batch_size=32,
                                                                spatial_dims=spatial_dims)
    
    # Post-processing: Replace negative pr values with zeros
    if var_target == 'pr':
        pred_da = pred_da.where(pred_da >= 0, 0)
    
    # Save predictions and ground truth
    os.makedirs(VALIDATION_PATH, exist_ok=True)
    
    # Create output dataset with predictions and ground truth
    output_ds = xr.Dataset({
        var_target: pred_da[var_target]
    })
    
    output_suffix = '-orog' if use_orography else ''
    output_name = f'DeepESD_{training_experiment}_{domain}_{var_target}{output_suffix}'
    pred_path = os.path.join(VALIDATION_PATH, f'{output_name}_predictions.nc')
    gt_path = os.path.join(VALIDATION_PATH, f'{output_name}_groundtruth.nc')
    
    print(f"Saving predictions to {pred_path}")
    output_ds.to_netcdf(pred_path)
    
    print(f"Saving ground truth to {gt_path}")
    y_test[[var_target]].to_netcdf(gt_path)
    
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python predict_validation.py <var_target> <domain> <training_experiment> [use_orography]")
        print("Example: python predict_validation.py pr ALPS ESD_pseudo_reality")
        print("Example: python predict_validation.py pr ALPS ESD_pseudo_reality true")
        sys.exit(1)
    
    var_target = sys.argv[1]
    domain = sys.argv[2]
    training_experiment = sys.argv[3]
    use_orography = sys.argv[4].lower() in ('true', '1', 'yes', 'on') if len(sys.argv) > 4 else False
    
    main(var_target, domain, training_experiment, use_orography)
