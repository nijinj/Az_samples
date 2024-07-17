# File: data_prep.py
from azure.ai.ml import Input, Output
from azure.ai.ml.dsl import pipeline

def prepare_data(input_data: Input(type="uri_file"),
                 output_data: Output(type="uri_file")):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(input_data)
    X = df.drop(['Survived'], axis=1)
    y = df['Survived']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = pd.concat([X_train, y_train], axis=1)
    train_data.to_csv(output_data, index=False)

# File: pipeline_definition.py
from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.automl import AutoMLClassificationJob
from data_prep import prepare_data

@pipeline(name="titanic_automl_pipeline",
          description="AutoML pipeline for Titanic dataset")
def titanic_pipeline(
    data_input,
    automl_compute_name: str,
    experiment_name: str = "titanic-automl-experiment"
):
    prepared_data = prepare_data(input_data=data_input)
    
    automl_job = AutoMLClassificationJob(
        compute=automl_compute_name,
        experiment_name=experiment_name,
        training_data=prepared_data.outputs.output_data,
        target_column_name="Survived",
        primary_metric="accuracy",
        n_cross_validations=5,
        enable_model_explainability=True,
        allowed_training_time_hours=0.5
    )
    
    return {"best_model": automl_job.outputs.best_model}

# File: main.py
from azure.ai.ml import MLClient, Input
from azure.identity import DefaultAzureCredential
from pipeline_definition import titanic_pipeline

def get_ml_client():
    return MLClient.from_config(credential=DefaultAzureCredential())

def main():
    ml_client = get_ml_client()
    
    input_data = Input(type="uri_file", path="azureml:titanic_dataset:1")
    
    pipeline_job = titanic_pipeline(
        data_input=input_data,
        automl_compute_name="your-compute-cluster-name"
    )
    
    returned_job = ml_client.jobs.create_or_update(pipeline_job)
    ml_client.jobs.stream(returned_job.name)
    
    best_model = returned_job.outputs["best_model"]
    registered_model = ml_client.models.create_or_update(best_model)
    print(f"Registered model: {registered_model.name}")

if __name__ == "__main__":
    main()
