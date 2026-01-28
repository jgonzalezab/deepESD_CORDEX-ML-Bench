"""Shared configuration for training and evaluation."""

# Experiment settings
EVALUATION_EXPERIMENT_SETTINGS = {
    "ESD_pseudo_reality": {
        "Perfect_cross_validation": ("historical", "perfect", True),
        "Imperfect_cross_validation": ("historical", "imperfect", True),
        "Perfect_extrapolation": ("mid_end_century", "perfect", True),
        "Imperfect_extrapolation": ("mid_end_century", "imperfect", True),
        "Perfect_extrapolation_GCM_transferability": ("mid_end_century", "perfect", False),
    },
    "Emulator_hist_future": {
        "Perfect_cross_validation": ("historical", "perfect", True),
        "Imperfect_cross_validation": ("historical", "imperfect", True),
        "Perfect_interpolation": ("mid_century", "perfect", True),
        "Imperfect_interpolation": ("mid_century", "imperfect", True),
        "Perfect_interpolation_GCM_transferability": ("mid_century", "perfect", False),
        "Imperfect_interpolation_GCM_transferability": ("mid_century", "imperfect", False),
    },
}

# Map periods to dates
PERIOD_DATES = {"historical": "1981-2000",
                "mid_century": "2041-2060",
                "end_century": "2080-2099",
                "mid_end_century": ["2041-2060", "2080-2099"]}

# GCM selection by domain and training setup
GCM_TRAIN = {"NZ": "ACCESS-CM2", "ALPS": "CNRM-CM5", "SA": "ACCESS-CM2"}
GCM_EVAL = {"NZ": "EC-Earth3", "ALPS": "MPI-ESM-LR", "SA": "NorESM2-MM"}

# Training hyperparameters
TRAINING_CONFIG = {"batch_size": 64,
                   "num_epochs": 10000,
                   "patience_early_stopping": 60,
                   "learning_rate": 0.0001,
                   "validation_split": 0.1}

# Paths
DATA_PATH = "/gpfs/projects/meteo/WORK/gonzabad/deepESD_CORDEX-ML-Bench/data/Bench-data"
MODEL_PATH = "/gpfs/projects/meteo/WORK/gonzabad/deepESD_CORDEX-ML-Bench/models"
SUBMISSION_PATH = "/gpfs/projects/meteo/WORK/gonzabad/deepESD_CORDEX-ML-Bench/data/submission"
TEMPLATES_PATH = "/gpfs/projects/meteo/WORK/gonzabad/ml-benchmark/format_predictions/templates"
ASYM_PATH = '/gpfs/projects/meteo/WORK/gonzabad/deepESD_CORDEX-ML-Bench/data/asym'
VALIDATION_PATH = "/gpfs/projects/meteo/WORK/gonzabad/deepESD_CORDEX-ML-Bench/data/validation"
FIGS_PATH = "/gpfs/projects/meteo/WORK/gonzabad/deepESD_CORDEX-ML-Bench/figs"

