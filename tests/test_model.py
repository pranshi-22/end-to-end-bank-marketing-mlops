
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_test_model():
    data = pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60],
        "job": [
            "admin.", "technician", "management", "admin.",
            "technician", "management", "admin.", "management"
        ],
        "y": [0, 0, 1, 0, 1, 1, 0, 1]
    })

    X = data.drop(columns=["y"])
    y = data["y"]

    numeric_features = ["age"]
    categorical_features = ["job"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42))
    ])

    model.fit(X, y)

    return model


def test_model_prediction():
    model = create_test_model()

    sample = pd.DataFrame({
        "age": [35],
        "job": ["management"]
    })

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_prediction_probability():
    model = create_test_model()

    sample = pd.DataFrame({
        "age": [35],
        "job": ["management"]
    })

    probability = model.predict_proba(sample)[0][1]

    assert 0 <= probability <= 1
    assert np.isfinite(probability)


def test_model_probability_shape():
    model = create_test_model()

    sample = pd.DataFrame({
        "age": [35],
        "job": ["management"]
    })

    probabilities = model.predict_proba(sample)

    assert probabilities.shape == (1, 2)
    assert np.isclose(probabilities.sum(), 1.0)
