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
from deep4downscaling.deep.models.deepesd import DeepESDpr, DeepESDtas

from config import MODEL_PATH, DATA_PATH, TRAINING_CONFIG, ASYM_PATH
from data_utils import (load_predictor_and_predictand, preprocess_data,
                        split_train_test, get_spatial_dims)


def main(var_target: str, domain: str, training_experiment: str):
    """Train the DeepESD model."""
    
    # Setup model and training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data for {domain} domain, {var_target} variable...")
    predictor, predictand = load_predictor_and_predictand(DATA_PATH, domain, training_experiment, var_target)
    
    # Preprocess
    predictor, predictand = preprocess_data(predictor, predictand, domain)
    
    # Split into train and test
    x_train, y_train, _, _ = split_train_test(predictor, predictand, training_experiment)
    
    # Standardize predictor
    x_train_standardized = deep4downscaling.trans.standardize(data_ref=x_train, data=x_train)
    
    # Stack spatial dimensions
    spatial_dims = get_spatial_dims(domain)
    y_train_stacked = y_train.stack(gridpoint=spatial_dims)
    
    # Convert to numpy
    x_train_arr = deep4downscaling.trans.xarray_to_numpy(x_train_standardized)
    y_train_arr = deep4downscaling.trans.xarray_to_numpy(y_train_stacked)
    
    # Set loss function
    if var_target == 'pr':
        loss_function = deep4downscaling.deep.loss.Asym(ignore_nans=False,
                                                        appendix=f'{training_experiment}_{domain}',
                                                        asym_path=ASYM_PATH)
        if loss_function.parameters_exist():
            loss_function.load_parameters()
        else:
            loss_function.compute_parameters(data=y_train_stacked,
                                             var_target=var_target)
        loss_function.prepare_parameters(device=device)
    else:
        loss_function = deep4downscaling.deep.loss.MseLoss(ignore_nans=False)
    
    # Create datasets and dataloaders
    train_dataset = deep4downscaling.deep.utils.StandardDataset(x=x_train_arr, y=y_train_arr)
    
    train_dataset, valid_dataset = random_split(train_dataset, 
                                                [1 - TRAINING_CONFIG['validation_split'],
                                                 TRAINING_CONFIG['validation_split']])
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=TRAINING_CONFIG['batch_size'],
                                  shuffle=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=TRAINING_CONFIG['batch_size'],
                                  shuffle=True)
    
    # Create model name
    model_name = f'DeepESD_{training_experiment}_{domain}_{var_target}'
    
    if var_target == 'pr':
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
    
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=TRAINING_CONFIG['learning_rate'])
    
    # Train
    print(f"Training model: {model_name}")
    train_loss, val_loss = deep4downscaling.deep.train.standard_training_loop(model=model,
                                                                              model_name=model_name,
                                                                              model_path=MODEL_PATH,
                                                                              device=device,
                                                                              num_epochs=TRAINING_CONFIG['num_epochs'],
                                                                              loss_function=loss_function,
                                                                              optimizer=optimizer,
                                                                              train_data=train_dataloader,
                                                                              valid_data=valid_dataloader,
                                                                              patience_early_stopping=TRAINING_CONFIG['patience_early_stopping'])
    print("Training completed!")

if __name__ == "__main__":
    var_target = sys.argv[1]
    domain = sys.argv[2]
    training_experiment = sys.argv[3]
    
    main(var_target, domain, training_experiment)