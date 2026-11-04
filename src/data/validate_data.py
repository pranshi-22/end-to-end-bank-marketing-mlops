import pandas as pd
from pathlib import Path


def validate_data(file_path: str) -> bool:
    """Validate the bank marketing dataset."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, sep=";")

    required_columns = [
        "age", "job", "marital", "education",
        "default", "housing", "loan", "contact",
        "month", "day_of_week", "duration",
        "campaign", "pdays", "previous", "poutcome",
        "emp.var.rate", "cons.price.idx",
        "cons.conf.idx", "euribor3m",
        "nr.employed", "y"
    ]

    if list(df.columns) != required_columns:
        raise ValueError("Dataset columns do not match expected schema.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["y"].isnull().any():
        raise ValueError("Target column contains missing values.")

    if not df["y"].isin(["yes", "no"]).all():
        raise ValueError("Invalid target values found.")

    print("Data validation successful!")
    print(f"Records: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return True


if __name__ == "__main__":
    validate_data("data/raw/bank-additional-full.csv")