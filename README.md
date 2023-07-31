# Eric Albrecht MLOps Zoomcamp Final Project: BikeShare Next Day Ride Count Predictions

## Problem Description
The original dataset is a bike rental dataset from the city of Austin Texas. The rental service has to place the bikes out every morning depending on what they project the customers for the day to be. If more bikes are placed out than are required, then they risk damage to the scooters or loss of money for paying people to transport the scooters out to the public. If not enough scooters are put out, they lose money from people wanting to rent but not having available supply. The company wants to optimize profit by placing the predicted amount of scooters needed to rent for the public, to minimize losses from either scenario. 

## Dataset: https://www.kaggle.com/datasets/jboysen/austin-bike?resource=download

## Dependency Version: 
See requirements.txt in the repository. Can directly be used to create conda environment and run the project.

## Functionality Covered
Pre-Commit Framework: Black (Formatter), Flake8 (Linter), Pytest (Code test suite)<br>
Unit Tests in tests/general_test.py: Model Only (test), Integration with Deployment<br>
Experiment Tracking: MLFlow used to track model parameters and metrics for comparison<br>
ML Model Registry: MLFlow used to host the model, and serve it to the deployment mechanisms.<br>
Local Deployment: Python script ml_model_deployment/deploy_model.py used to launch endpoint, and the integration test in general_tests.py was used to validate this.<br>
Containerized Deployment: The Dockerfile contains the recipe to build a container, then run the Docker image. General_tests can be used for this as well. <br>
Cloud/Emulator: Was not used for this project<br>
Best Practices: Explainable, formal data engineering pipeline and feature storage for analysis<br>
Model Monitoring: Evidently Test Suite for Data Drift on Model Input. Report output to data/drift_reference with unique uuid numbers for saving.<br>
CI/CD Pipeline: Was not used for this project.<br>

## Prefect Flows to Monitor at localhost:4200
Data Pipeline Flow: Performs the standard data engineering for medallion architecture to get data in.<br>
ML Experiment Flow: Performs the ML Experiment and contributes any new best model to the MLFlow Registry. <br>
This project provides full workflow pipeline from raw data entry files to output best model based on training data. Serving utilities are then run separately.<br>

## MLFlow Experiments to Register the Model
Ride Count Non-Holiday
****Note: This should be the first experiment in your MLFlow, aka experiment_id = 1.

## Server Initializations to Support (before anything can be run)
Prefect: prefect server start <br>
MLFlow: mlflow server --backend-store-uri sqlite:///ml.db --artifacts-destination .\ml_experiments\mlflow_artifacts\<br>
Containerized App from Python: docker run model_deploy<br>
Python Script from local (Dev only): cd ml_model_deployment && uvicorn deploy_model:app --reload<br>

## Order to Run Items
Download the dataset from the above link. Place both CSV’s in the data/raw/directory. The pipeline will transform and convert the data formats.<br>
Create a conda environment using the requirements.txt as the foundation. Conda create -n requirements.txt.<br>
Configure all servers through initialization. Make sure to be in the parent directory or project when doing this.<br>
Create the MLFlow Experiment, if it does not exist.<br>
From the parent directory, run main.py. This will generate the ML Models, and place the best one into the Model registry for the registered model name. <br>
Testing the API: Follow the above instructions for the local python script, or run docker build ., rename the model by image id, and deploy the model. The above script should be able to test the build as well. NOTE: make sure to turn deployed flag to True, when testing these. IN ORDER TO RUN THE MODEL UNIT TESTS, a model has to be generated and available in the MLFlow registry for usage.

## To Run All Tests
Make sure the MLFlow server is running at the 5000 Port address on localhost.<br>
The Live Deployment Test must have the deploy_model.py script running and MLFlow Server up, and the deployed flag turned on.
