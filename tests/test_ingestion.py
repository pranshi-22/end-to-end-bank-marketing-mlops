
import pandas as pd

from src.data.ingest_data import load_data


def create_test_dataset(tmp_path):
    data = pd.DataFrame({
        "age": [35, 42, 29],
        "job": ["management", "technician", "admin."],
        "y": ["yes", "no", "yes"]
    })

    file_path = tmp_path / "test_dataset.csv"
    data.to_csv(file_path, index=False, sep=";")

    return file_path


def test_data_loading(tmp_path):
    file_path = create_test_dataset(tmp_path)

    df = load_data(str(file_path))

    assert df is not None
    assert not df.empty
    assert df.shape == (3, 3)


def test_target_column_exists(tmp_path):
    file_path = create_test_dataset(tmp_path)

    df = load_data(str(file_path))

    assert "y" in df.columns


def test_target_values_are_valid(tmp_path):
    file_path = create_test_dataset(tmp_path)

    df = load_data(str(file_path))

    assert set(df["y"].unique()) == {"yes", "no"}
