import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.features.build_features import build_preprocessor


def train_with_mlflow():

    mlflow.set_experiment("Bank-Marketing-Prediction")

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df.drop(columns=["y"])
    y_train = train_df["y"]

    X_test = test_df.drop(columns=["y"])
    y_test = test_df["y"]

    models = {
        "Logistic_Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    for model_name, classifier in models.items():

        with mlflow.start_run(run_name=model_name):

            preprocessor = build_preprocessor(X_train)

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ])

            pipeline.fit(X_train, y_train)
            input_example = X_train.head(3).copy()

            integer_columns = input_example.select_dtypes(
                include=["int64", "int32"]
            ).columns

            input_example[integer_columns] = input_example[integer_columns].astype("float64")

            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            signature_input = X_test.copy()

            integer_columns = signature_input.select_dtypes(
                include=["int64", "int32"]
            ).columns

            signature_input[integer_columns] = signature_input[
                integer_columns
            ].astype("float64")

            signature = mlflow.models.infer_signature(
                signature_input,
                predictions
            )

            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "f1_score": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "roc_auc": roc_auc_score(y_test, probabilities)
            }

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_param("dataset", "UCI Bank Marketing")
            mlflow.log_param("target_column", "y")
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("preprocessing", "Median imputation + StandardScaler + OneHotEncoder")
            mlflow.log_param("excluded_feature", "duration")
            mlflow.log_param("training_rows", len(X_train))
            mlflow.log_param("testing_rows", len(X_test))

            if model_name == "Random_Forest":
                mlflow.log_param("n_estimators", 200)
            else:
                mlflow.log_param("max_iter", 2000)

            mlflow.set_tag("project", "Bank Marketing Prediction")
            mlflow.set_tag("task", "Binary Classification")
            mlflow.set_tag("prediction_stage", "Pre-call")
            mlflow.set_tag("experiment_type", "Baseline Model")
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=signature,
                input_example=input_example,
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree"
                ]
            )

            print(f"\n{model_name}")
            print(metrics)
            print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    train_with_mlflow()