import pandas as pd


def compare_models():
    """Compare trained models using evaluation metrics."""

    results_path = "models/model_results.csv"

    results = pd.read_csv(results_path)

    results = results.sort_values(
        by="f1_score",
        ascending=False
    )

    print("\nMODEL COMPARISON")
    print("=" * 70)
    print(results.to_string(index=False))

    best_model = results.iloc[0]

    print("\nBest model based on F1-score:")
    print(best_model["model"])
    print(f"F1-score: {best_model['f1_score']:.4f}")
    print(f"ROC-AUC: {best_model['roc_auc']:.4f}")


if __name__ == "__main__":
    compare_models()