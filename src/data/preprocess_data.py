import pandas as pd
from pathlib import Path


def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """Clean and prepare the bank marketing dataset."""

    df = pd.read_csv(input_path, sep=";")

    original_records = len(df)

    # Remove exact duplicate records
    df = df.drop_duplicates().copy()

    # Remove duration to prevent data leakage
    df = df.drop(columns=["duration"])

    # Separate features and target
    X = df.drop(columns=["y"])
    y = df["y"].map({"no": 0, "yes": 1})

    # Combine features and target
    processed_df = X.copy()
    processed_df["y"] = y

    # Save processed dataset
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output, index=False)

    print("Preprocessing completed successfully!")
    print(f"Original records: {original_records}")
    print(f"Records after cleaning: {len(processed_df)}")
    print(f"Duplicates removed: {original_records - len(processed_df)}")
    print(f"Features: {X.shape[1]}")
    print(f"Target distribution:\n{y.value_counts()}")

    return processed_df


if __name__ == "__main__":
    preprocess_data(
        "data/raw/bank-additional-full.csv",
        "data/processed/bank_marketing_processed.csv"
    )