from src.data.ingest_data import load_data


def test_data_loading():
    df = load_data("data/raw/bank-additional-full.csv")

    assert df is not None
    assert not df.empty
    assert df.shape == (41188, 21)


def test_target_column_exists():
    df = load_data("data/raw/bank-additional-full.csv")

    assert "y" in df.columns


def test_target_values_are_valid():
    df = load_data("data/raw/bank-additional-full.csv")

    assert set(df["y"].unique()) == {"yes", "no"}