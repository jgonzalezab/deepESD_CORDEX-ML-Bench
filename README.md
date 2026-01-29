# DeepESD CORDEX-ML-Bench

This repository contains the implementation of the DeepESD model for the CORDEX ML-Benchmark. It includes scripts for training, validation, evaluation, and generating submission files.

## Workflow

The project follows a two-phase workflow: 
1. **Validation Phase**: Train on a subset of data and evaluate performance on a reserved "fake test" partition.
2. **Submission Phase**: Train on the full dataset and generate predictions for the benchmark.

### 1. Training (Validation Mode)

By default, the training script uses a "validation mode" that reserves a portion of the training data (e.g., the last year) for evaluation.

```bash
# Usage: python scripts/train.py <var_target> <domain> <training_experiment>
python scripts/train.py pr ALPS ESD_pseudo_reality
```

### 2. Validation Prediction

Once the model is trained, generate predictions on the reserved validation set to check performance.

```bash
# Usage: python scripts/predict_validation.py <var_target> <domain> <training_experiment>
python scripts/predict_validation.py pr ALPS ESD_pseudo_reality
```
*This saves predictions and ground truth files to `data/validation/`.*

### 3. Evaluation and Reporting

Run the evaluation suite to generate a comprehensive PDF report with spatial maps, boxplots, and spectral analysis.

The evaluation script can be configured via environment variables or by editing the script:

```bash
# Using environment variables (recommended)
VAR_TARGET=pr DOMAIN=ALPS TRAINING_EXPERIMENT=ESD_pseudo_reality python eval/run_all_eval.py
```

Alternatively, you can open `eval/run_all_eval.py` and configure the `params` dictionary directly. The resulting PDF will be saved in the `figs/` directory.

### 4. Batch Processing (Automation)

For running both prediction and evaluation for multiple variables, domains, or experiments at once, use the batch processing script:

```bash
# Run for multiple configurations
python scripts/run_batch_eval.py --vars pr tas --domains ALPS NZ --exps ESD_pseudo_reality
```

This script will sequentially run `predict_validation.py` and `run_all_eval.py` for each combination of the specified parameters.

### 5. Final Training (Full Dataset)

Once you are satisfied with the model's performance, modify `scripts/train.py` to use 100% of the training data:

```python
# In scripts/train.py, change validation_mode to False
x_train, y_train, _, _ = split_train_test(predictor, predictand, training_experiment, validation_mode=False)
```

Then retrain the model and run the submission script:

```bash
python scripts/submission.py
```

## Repository Structure

- `scripts/`: Main execution scripts.
    - `train.py`: Model training script.
    - `predict_validation.py`: Generates predictions on the validation set.
    - `run_batch_eval.py`: Automates prediction and evaluation for multiple configs.
    - `submission.py`: Generates the final submission ZIP for the benchmark.
- `eval/`: Diagnostic and reporting scripts.
    - `run_all_eval.py`: Master script to run all evaluations and merge PDFs.
    - `standard_metrics.py`: Spatial bias and RMSE metrics.
    - `psd_comparison.py`: Power Spectral Density analysis.
    - `daily_comparison.py`: Visual comparison of individual days.
- `src/`: Core logic and configuration.
    - `config.py`: Path and hyperparameter settings.
    - `data_utils.py`: Data loading and splitting logic.
- `data/`: Data storage (NetCDF files).
- `models/`: Saved model weights (`.pt` files).
- `figs/`: Generated evaluation reports.

## Requirements

The evaluation scripts require `pypdf` or `PyPDF2` for merging reports:
```bash
pip install pypdf
```
The project depends on the `deep4downscaling` library and the `ml-benchmark` repository being available in the parent or specified directories.
