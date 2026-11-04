import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_data(input_path: str, output_dir: str):
    """Split the processed dataset into training and testing sets."""

    df = pd.read_csv(input_path)

    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_df = X_train.copy()
    train_df["y"] = y_train

    test_df = X_test.copy()
    test_df["y"] = y_test

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_path / "train.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)

    print("Train-test split completed!")
    print(f"Training records: {len(train_df)}")
    print(f"Testing records: {len(test_df)}")
    print(f"Training target distribution:\n{y_train.value_counts()}")
    print(f"Testing target distribution:\n{y_test.value_counts()}")


if __name__ == "__main__":
    split_data(
        "data/processed/bank_marketing_processed.csv",
        "data/processed"
    )