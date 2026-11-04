import pandas as pd
from pathlib import Path


def load_data(file_path: str) -> pd.DataFrame:
    """Load the bank marketing dataset."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, sep=";")

    print(f"Dataset loaded successfully: {df.shape}")
    return df


if __name__ == "__main__":
    data_path = "data/raw/bank-additional-full.csv"
    df = load_data(data_path)

    print(df.head())