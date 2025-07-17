from fastapi import FastAPI
from models import download_data, prepare_data, build_model, train_and_evaluate
import numpy as np

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Stock Forecasting API!"}

@app.get("/forecast")
def run_forecast():
    try:
        stock_data = download_data()
        X_train, X_test, y_train, y_test, scaler = prepare_data(stock_data)

        results = {}
        model_names = ['LSTM', 'CNN', 'ANN', 'RNN', 'GRU']

        for name in model_names:
            model = build_model(name)
            result = train_and_evaluate(model, X_train, y_train, X_test, y_test, scaler, name)
            # For now, we just store metrics
            results[name] = {
                "rmse": float(np.round(result["history"].history["val_loss"][-1] ** 0.5, 4))
            }

        return {"models": results}

    except Exception as e:
        return {"error": str(e)}
