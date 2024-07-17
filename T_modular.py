import pandas as pd
from sklearn.model_selection import train_test_split
from azure.ai.ml import MLClient
from azure.ai.ml.automl import AutoMLClassification
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from azure.ai.ml import Input
import argparse
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AutoML job on Titanic dataset")
    parser.add_argument("--dataset", type=str, default="titanic_dataset", help="Name of the dataset in Azure ML workspace")
    parser.add_argument("--compute", type=str, required=True, help="Name of the compute cluster")
    parser.add_argument("--experiment", type=str, default="titanic-automl-experiment", help="Name of the experiment")
    return parser.parse_args()

def get_ml_client():
    try:
        return MLClient.from_config(credential=DefaultAzureCredential())
    except Exception as e:
        logging.error(f"Failed to connect to Azure ML workspace: {e}")
        raise

def load_and_prepare_data(ml_client, dataset_name):
    try:
        titanic_data = ml_client.data.get(dataset_name)
        df = pd.read_csv(titanic_data.path)
        X = df.drop(['Survived'], axis=1)
        y = df['Survived']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        train_data = pd.concat([X_train, y_train], axis=1)
        return train_data
    except Exception as e:
        logging.error(f"Failed to load and prepare data: {e}")
        raise

def create_data_asset(ml_client, train_data):
    try:
        train_data.to_csv("train_data.csv", index=False)
        return ml_client.data.create_or_update(
            Input(type=AssetTypes.URI_FILE, path="./train_data.csv")
        )
    except Exception as e:
        logging.error(f"Failed to create data asset: {e}")
        raise

def configure_automl_job(compute_name, experiment_name, train_data_asset):
    return AutoMLClassification(
        compute=compute_name,
        experiment_name=experiment_name,
        training_data=train_data_asset,
        target_column_name="Survived",
        primary_metric="accuracy",
        n_cross_validations=5,
        enable_model_explainability=True,
        allowed_training_time_hours=0.5
    )

def run_automl_job(ml_client, automl_job):
    try:
        returned_job = ml_client.jobs.create_or_update(automl_job)
        ml_client.jobs.stream(returned_job.name)
        return ml_client.jobs.get(returned_job.name).outputs.best_model
    except Exception as e:
        logging.error(f"Failed to run AutoML job: {e}")
        raise

def register_model(ml_client, best_model):
    try:
        registered_model = ml_client.models.create_or_update(best_model)
        logging.info(f"Registered model: {registered_model.name}")
    except Exception as e:
        logging.error(f"Failed to register model: {e}")
        raise

def main():
    setup_logging()
    args = parse_arguments()
    
    ml_client = get_ml_client()
    train_data = load_and_prepare_data(ml_client, args.dataset)
    train_data_asset = create_data_asset(ml_client, train_data)
    
    automl_job = configure_automl_job(args.compute, args.experiment, train_data_asset)
    best_model = run_automl_job(ml_client, automl_job)
    
    logging.info(f"Best model: {best_model}")
    register_model(ml_client, best_model)

if __name__ == "__main__":
    main()
