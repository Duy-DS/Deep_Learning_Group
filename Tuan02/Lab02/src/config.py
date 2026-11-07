"""
Centralized Configuration and Path Management for Lab02
Provides dynamic path resolution regardless of working directory.
"""
from pathlib import Path

# Thư mục gốc của Lab02
LAB02_ROOT = Path(__file__).resolve().parents[1]

# Thư mục dữ liệu
DATA_DIR = LAB02_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" if (DATA_DIR / "raw").exists() else DATA_DIR

# Thư mục dữ liệu đã xử lý
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Thư mục artifacts, models, outputs
MODELS_DIR = LAB02_ROOT / "models"
OUTPUTS_DIR = LAB02_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PREPROCESSING_OUTPUT_DIR = OUTPUTS_DIR / "preprocessing"

# Hyperparameters mặc định
RANDOM_STATE = 42
TEST_SIZE = 0.2
BATCH_SIZE = 32
DEFAULT_INPUT_DIM = 314
TARGET_COLUMN = "SalePrice"
TARGET_LOG_COLUMN = "SalePrice_log"
ID_COLUMN = "Id"
SPLIT_COLUMN = "split"
