import pandas as pd
from sklearn.model_selection import train_test_split
from azure.ai.ml import MLClient
from azure.ai.ml.automl import AutoMLClassification, AutoMLJob
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from azure.ai.ml import Input

# Connect to your Azure ML workspace
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# Load the Titanic dataset
# Assuming the dataset is already uploaded to the default datastore
titanic_data = ml_client.data.get("titanic_dataset")
df = pd.read_csv(titanic_data.path)

# Prepare the data
X = df.drop(['Survived'], axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine features and target for training
train_data = pd.concat([X_train, y_train], axis=1)

# Save the training data to a CSV file
train_data.to_csv("train_data.csv", index=False)

# Upload the training data to the datastore
train_data_asset = ml_client.data.create_or_update(
    Input(type=AssetTypes.URI_FILE, path="./train_data.csv")
)

# Define AutoML job
automl_classification_job = AutoMLClassification(
    compute="your-compute-cluster-name",
    experiment_name="titanic-automl-experiment",
    training_data=train_data_asset,
    target_column_name="Survived",
    primary_metric="accuracy",
    n_cross_validations=5,
    enable_model_explainability=True,
    allowed_training_time_hours=0.5
)

# Submit the AutoML job
returned_job = ml_client.jobs.create_or_update(automl_classification_job)

# Wait for the job to complete
ml_client.jobs.stream(returned_job.name)

# Get the best model
best_model = ml_client.jobs.get(returned_job.name).outputs.best_model

# Print the best model details
print(best_model)

# Optionally, register the model
registered_model = ml_client.models.create_or_update(best_model)
print(f"Registered model: {registered_model.name}")
