import mlflow.pyfunc

MODEL_NAME = "Bank-Marketing-Subscription-Predictor"

def load_champion_model():
    model_uri = f"models:/{MODEL_NAME}@champion"

    model = mlflow.pyfunc.load_model(model_uri)

    print("Champion model loaded successfully!")

    return model


if __name__ == "__main__":
    model = load_champion_model()