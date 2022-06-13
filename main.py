import pickle

from model import train as model_train
from model import predict as model_predict

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI(title="House Project")

@app.post("/train")
def train():
    model_train()
    return {"status": "success"}

@app.post("/predict")
def predict():
    predictions, mae, mse, rmse, r2_square = model_predict()
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2_square": r2_square,
        "response": predictions.tolist()
    }

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
         title="HOME SWEET HOME",
        version="1",
        description="Home Sweet Home helps you estimate your house or your dream's house prices. Project by: Beyza Kerçek , Elif Yetkinoğlu, Gülben Süslü ",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

f = open("intro.txt", "r")
print(f.read())
