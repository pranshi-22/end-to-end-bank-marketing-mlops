
import pandas as pd

from src.data.validate_data import validate_data


def create_valid_dataset():
    return pd.DataFrame({
        "age": [35, 42, 29],
        "job": ["management", "technician", "admin."],
        "marital": ["married", "single", "married"],
        "education": [
            "university.degree",
            "professional.course",
            "high.school"
        ],
        "default": ["no", "no", "unknown"],
        "housing": ["yes", "no", "yes"],
        "loan": ["no", "no", "yes"],
        "contact": ["cellular", "telephone", "cellular"],
        "month": ["may", "jun", "jul"],
        "day_of_week": ["mon", "tue", "wed"],
        "duration": [120, 200, 150],
        "campaign": [1, 2, 1],
        "pdays": [999, 999, 3],
        "previous": [0, 0, 1],
        "poutcome": ["nonexistent", "nonexistent", "success"],
        "emp.var.rate": [1.1, 1.4, -0.1],
        "cons.price.idx": [93.994, 94.465, 93.2],
        "cons.conf.idx": [-36.4, -41.8, -42.0],
        "euribor3m": [4.857, 4.865, 4.2],
        "nr.employed": [5191.0, 5228.1, 5195.0],
        "y": ["yes", "no", "yes"]
    })


def test_dataset_validation(tmp_path):
    dataset_path = tmp_path / "test_dataset.csv"

    create_valid_dataset().to_csv(
        dataset_path, index=False, sep=";"
    )

    assert validate_data(str(dataset_path)) is True


def test_processed_dataset_creation(tmp_path):
    output_path = tmp_path / "bank_marketing_processed.csv"

    create_valid_dataset().to_csv(output_path, index=False)

    assert output_path.exists()


def test_train_test_datasets_creation(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    create_valid_dataset().iloc[:2].to_csv(train_path, index=False)
    create_valid_dataset().iloc[2:].to_csv(test_path, index=False)

    assert train_path.exists()
    assert test_path.exists()
