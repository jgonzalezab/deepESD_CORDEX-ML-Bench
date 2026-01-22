"""Shared configuration for training and evaluation."""

# Experiment settings
EVALUATION_EXPERIMENT_SETTINGS = {
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

