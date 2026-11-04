import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib


def build_preprocessor(X: pd.DataFrame):
    """Build preprocessing pipelines for numerical and categorical features."""

    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "str"]).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return preprocessor


if __name__ == "__main__":
    train_path = "data/processed/train.csv"

    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=["y"])

    preprocessor = build_preprocessor(X_train)

    preprocessor.fit(X_train)

    output_path = Path("models/preprocessor.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, output_path)

    print("Feature preprocessing pipeline fitted successfully!")
    print(f"Numerical features: {X_train.select_dtypes(include='number').shape[1]}")
    print(f"Categorical features: {X_train.select_dtypes(include=['object', 'str']).shape[1]}")
    print(f"Preprocessor saved at: {output_path}")