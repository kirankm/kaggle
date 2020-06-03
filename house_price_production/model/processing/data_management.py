def load_dataset(*, file_name: str) -> pd.DataFrame:
    return pd.read_csv(f'{config.DATASET_PATH/file_name}')