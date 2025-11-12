"""Training script for DeepESD model."""

import sys
import torch
from torch.utils.data import DataLoader, random_split

sys.path.append("/gpfs/projects/meteo/WORK/gonzabad/deep4downscaling")
sys.path.append("/gpfs/projects/meteo/WORK/gonzabad/deepESD_CORDEX-ML-Bench/src")

import deep4downscaling.trans
import deep4downscaling.deep.loss
import deep4downscaling.deep.utils
import deep4downscaling.deep.train

import models
from config import MODEL_PATH, DATA_PATH, TRAINING_CONFIG
from data_utils import (
    load_predictor_and_predictand, load_orography, preprocess_data,
    split_train_test, get_spatial_dims
)


def main(var_target: str, use_orography: str, domain: str, training_experiment: str):
    """Train the DeepESD model."""
    
    # Convert orography flag to boolean
    bool_map = {"True": True, "False": False, "true": True, "false": False}
    use_orog = bool_map.get(use_orography)
    
    # Load data
    print(f"Loading data for {domain} domain, {var_target} variable...")
    predictor, predictand = load_predictor_and_predictand(
        DATA_PATH, domain, training_experiment, var_target
    )
    orog_data = load_orography(DATA_PATH, domain, training_experiment, use_orog)
    
    # Preprocess
    predictor, predictand, orog_data_stand = preprocess_data(
        predictor, predictand, domain, orog_data
    )
    
    # Split into train and test
    x_train, y_train, _, _ = split_train_test(predictor, predictand, training_experiment)
    
    # Standardize predictor
    x_train_standardized = deep4downscaling.trans.standardize(data_ref=x_train, data=x_train)
    
    # Stack spatial dimensions
    spatial_dims = get_spatial_dims(domain)
    y_train_stacked = y_train.stack(gridpoint=spatial_dims)
    orog_data_stacked = orog_data_stand.stack(gridpoint=spatial_dims) if orog_data_stand is not None else None
    
    # Convert to numpy
    x_train_arr = deep4downscaling.trans.xarray_to_numpy(x_train_standardized)
    y_train_arr = deep4downscaling.trans.xarray_to_numpy(y_train_stacked)
    orog_data_arr = deep4downscaling.trans.xarray_to_numpy(orog_data_stacked) if orog_data_stacked is not None else None
    
    # Set loss function
    loss_function = deep4downscaling.deep.loss.MseLoss(ignore_nans=False)
    if var_target == 'pr':
        print('WARNING: Using MSE loss for precipitation. Consider alternative loss functions.')
    
    # Create datasets and dataloaders
    train_dataset = deep4downscaling.deep.utils.StandardDataset(x=x_train_arr, y=y_train_arr)
    train_dataset, valid_dataset = random_split(
        train_dataset, 
        [1 - TRAINING_CONFIG['validation_split'], TRAINING_CONFIG['validation_split']]
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True
    )
    
    # Setup model and training
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
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate']
    )
    
    # Train
    print(f"Training model: {model_name}")
    train_loss, val_loss = deep4downscaling.deep.train.standard_training_loop(
        model=model,
        model_name=model_name,
        model_path=MODEL_PATH,
        device=device,
        num_epochs=TRAINING_CONFIG['num_epochs'],
        loss_function=loss_function,
        optimizer=optimizer,
        train_data=train_dataloader,
        valid_data=valid_dataloader,
        patience_early_stopping=TRAINING_CONFIG['patience_early_stopping']
    )
    print("Training completed!")


if __name__ == "__main__":
    var_target = sys.argv[1]
    use_orography = sys.argv[2]
    domain = sys.argv[3]
    training_experiment = sys.argv[4]
    
    main(var_target, use_orography, domain, training_experiment)
