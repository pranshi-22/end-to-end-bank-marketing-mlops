import pandas as pd
import joblib


def test_trained_models_exist():
    import os

    assert os.path.exists("models/logistic_regression.joblib")
    assert os.path.exists("models/random_forest.joblib")
    assert os.path.exists("models/tuned_logistic_regression.joblib")


def test_model_prediction():
    model = joblib.load("models/tuned_logistic_regression.joblib")

    test_df = pd.read_csv("data/processed/test.csv")
    sample = test_df.drop(columns=["y"]).iloc[[0]]

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_prediction_probability():
    model = joblib.load("models/tuned_logistic_regression.joblib")

    test_df = pd.read_csv("data/processed/test.csv")
    sample = test_df.drop(columns=["y"]).iloc[[0]]

    probability = model.predict_proba(sample)[0][1]

    assert 0 <= probability <= 1