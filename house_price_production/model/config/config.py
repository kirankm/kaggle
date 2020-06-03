import pathlib

# filepaths
PATH = '/mnt/data/study_path/kaggle/house_price_production/model/'
SOURCE_PATH = pathlib.Path(PATH)
DATASET_PATH = SOURCE_PATH / "datasets"
TRAINED_MODEL_PATH = SOURCE_PATH / "trained_models"

# datasets
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
TARGET = "SalePrice"

# model parameters
RANDOM_STATE = 42


# log parameters
LOG_LEVEL = "warning"

TEST_VAR = 0