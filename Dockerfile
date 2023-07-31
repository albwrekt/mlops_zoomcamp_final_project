# Example container for launching the local containers

# Python installation base 3.11 image
FROM python:3.11-slim-bullseye

# Create working directory
WORKDIR /app

# COPY over the necessary files
COPY ml_model_deployment/deploy_model.py deploy_model.py
COPY utilities.py utilities.py
COPY deploy_requirements.txt deploy_requirements.txt

# Run pip install on the imported requirements.
RUN pip3 install -r deploy_requirements.txt

# Expose the port for FAST API
EXPOSE 8000

# Deploy and start the model the model
CMD ["uvicorn", "deploy_model:app", "--reload"]
