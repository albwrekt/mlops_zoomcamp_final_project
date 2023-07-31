"""
This file works to deploy the model as an API Endpoint for inference.
"""
# Library for API Endpoint
from fastapi import FastAPI, Request

# Data manipulation
import pandas as pd

# Add the system path
import sys

sys.path.insert(0, "../")
sys.path.insert(0, "../tests/")
# Import utilities
from utilities import *

# MLOps utilities
from mlflow import MlflowClient
import mlflow.pyfunc

mlflow.set_tracking_uri(mlflow_tracking_uri)

# make the app instance
app = FastAPI()


# Function leveraged in API.
def predict(features_df: pd.DataFrame):
    # Client to retrieve the model version
    client = MlflowClient(mlflow_tracking_uri)
    # Get the latest version
    latest_version = client.get_latest_versions(
        name=ride_count_non_holiday_model_name, stages=["Production"]
    )[0]
    # Load the model for predictions in the API
    model = mlflow.pyfunc.load_model(f"runs:/{latest_version.run_id}/model")
    # Make the predictions
    prediction = model.predict(features_df)
    # Return the prediction
    return prediction


# First method for processing a prediction.
@app.post("/")
async def ml_prediction(request: Request):
    """
    Return an ML prediction for the desired model.
    """
    # Translate the incoming request to a DataFrame
    result = await request.json()
    print(result)
    result = {key: [result[key]] for key in result.keys()}
    feature_df = pd.DataFrame.from_dict(result, orient="columns")
    print(feature_df)
    # Get the latest model and make a prediction
    prediction = predict(feature_df)
    print(prediction)
    # Package up the response
    response = {"Prediction": prediction[0]}
    response = {"he": "she"}
    # Return
    return response


# main method testing
if __name__ == "__main__":
    predict()
