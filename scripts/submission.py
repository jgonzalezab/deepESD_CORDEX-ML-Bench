"""
Submission script for the CORDEX ML-Benchmark for DeepESD.
"""

import os
import sys
import xarray as xr
import numpy as np
import torch
import zipfile
import glob

# Add source directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append("/gpfs/projects/meteo/WORK/gonzabad/deep4downscaling")

import deep4downscaling.trans
import deep4downscaling.deep.pred
from deep4downscaling.deep.models.deepesd import DeepESDpr, DeepESDtas

from config import MODEL_PATH, DATA_PATH, SUBMISSION_PATH, TEMPLATES_PATH
from data_utils import load_predictor_and_predictand, preprocess_data, split_train_test

# Set the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mapping from domain to info
DOMAIN_INFO = {'ALPS': {'spatial_dims': ('x', 'y')},
               'NZ': {'spatial_dims': ('lat', 'lon')},
               'SA': {'spatial_dims': ('lat', 'lon')}}

# Experiments to run
EXPERIMENTS = ['ESD_pseudo_reality', 'Emulator_hist_future']

def run_prediction(domain, experiment, predictor_path):
    """Computes the predictions for a specific predictor file."""
    ds_test = xr.open_dataset(predictor_path)
    if domain == 'SA': 
        ds_test = ds_test.drop_vars('time_bnds', errors='ignore')
    
    # Initialize output dataset
    ds_out = xr.Dataset(coords={'time': ds_test.time})
    spatial_dims = DOMAIN_INFO[domain]['spatial_dims']

    # Iterate over variables
    for var in ['pr', 'tasmax']:
        print(f"  Predicting {var}...")
        
        # Load training data to get standardization statistics
        predictor_train, predictand_train = load_predictor_and_predictand(DATA_PATH, domain, experiment, var)
        
        # Preprocess predictor_train to get the same days as used in training
        predictor_train, _ = preprocess_data(predictor_train, predictand_train, domain)
        x_train, y_train, _, _ = split_train_test(predictor_train, predictand_train, experiment)

        # Standardize test predictor using training statistics
        ds_test_stand = deep4downscaling.trans.standardize(data_ref=x_train, data=ds_test)
        
        # Get shapes for model initialization
        x_train_standardized = deep4downscaling.trans.standardize(data_ref=x_train, data=x_train)
        x_train_arr = deep4downscaling.trans.xarray_to_numpy(x_train_standardized)
        y_train_stacked = y_train.stack(gridpoint=spatial_dims)
        y_train_arr = deep4downscaling.trans.xarray_to_numpy(y_train_stacked)

        # Initialize and load model
        if var == 'pr':
            model = DeepESDpr(x_shape=x_train_arr.shape,
                              y_shape=y_train_arr.shape,
                              filters_last_conv=1,
                              stochastic=False,
                              last_relu=False)
        else:
            model = DeepESDtas(x_shape=x_train_arr.shape,
                               y_shape=y_train_arr.shape,
                               filters_last_conv=1,
                               stochastic=False)
        
        model_name = f'DeepESD_{experiment}_{domain}_{var}.pt'
        model_path = os.path.join(MODEL_PATH, model_name)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=DEVICE))
        model.eval()
        
        # Compute predictions using deep4downscaling
        y_mask = xr.ones_like(y_train.isel(time=0))
        pred_da = deep4downscaling.deep.pred.compute_preds_standard(x_data=ds_test_stand,
                                                                    model=model,
                                                                    device=DEVICE,
                                                                    var_target=var,
                                                                    mask=y_mask,
                                                                    batch_size=32,
                                                                    spatial_dims=spatial_dims)

        # Post-processing: Replace negative pr values with zeros
        if var == 'pr':
            pred_da = pred_da.where(pred_da >= 0, 0)

        # Load template for attributes
        template_path = os.path.join(TEMPLATES_PATH, f'{var}_{domain}.nc')
        ds_template = xr.open_dataset(template_path)
        
        # Ensure coordinates and attributes match template
        pred_da.attrs = ds_template[var].attrs
        ds_out[var] = pred_da

    return ds_out

if __name__ == "__main__":
    # Create the output base directory
    os.makedirs(SUBMISSION_PATH, exist_ok=True)

    # Iterate over the domains
    for domain in DOMAIN_INFO:
        print(f"Processing Domain: {domain}")
        
        # Find all test predictor files for this domain
        test_dir = f'{DATA_PATH}/{domain}/test'
        predictor_files = glob.glob(f'{test_dir}/**/*.nc', recursive=True)
        
        for pred_path in predictor_files:
            parts = pred_path.split(os.sep)
            try:
                test_idx = parts.index('test')
                period_folder = parts[test_idx + 1]
                condition = parts[test_idx + 3]
                filename = parts[-1]
            except (ValueError, IndexError):
                print(f"Skipping file with unexpected path structure: {pred_path}")
                continue
            
            # Iterate over the experiments
            for experiment in EXPERIMENTS:
                print(f"Predicting {filename} for {experiment} in {domain}...")
                
                ds_preds = run_prediction(domain, experiment, pred_path)
                
                training_label = experiment.replace("ESD_pseudo_reality", "ESD_Pseudo-Reality").replace("Emulator_hist_future", "Emulator_Hist_Future")
                
                out_dir = os.path.join(SUBMISSION_PATH, f"{domain}_Domain", training_label, period_folder, condition)
                os.makedirs(out_dir, exist_ok=True)
                
                out_filename = f"Predictions_pr_tasmax_{filename}"
                ds_preds.to_netcdf(os.path.join(out_dir, out_filename))

    # ZIP the submission
    zip_filename = "submission.zip"
    zip_path = os.path.join(SUBMISSION_PATH, zip_filename)

    print(f"Creating submission package: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(SUBMISSION_PATH):
            if zip_filename in files: continue # Don't include the zip itself
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, SUBMISSION_PATH)
                zipf.write(abs_path, rel_path)

    print("Done!")
