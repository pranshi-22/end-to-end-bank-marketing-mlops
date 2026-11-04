from src.data.validate_data import validate_data


def test_dataset_validation():
    result = validate_data("data/raw/bank-additional-full.csv")
    assert result is True


def test_processed_dataset_exists():
    import os

    assert os.path.exists(
        "data/processed/bank_marketing_processed.csv"
    )


def test_train_test_datasets_exist():
    import os

    assert os.path.exists("data/processed/train.csv")
    assert os.path.exists("data/processed/test.csv")