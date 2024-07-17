import pandas as pd
from sklearn.datasets import fetch_openml
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

# Connect to Azure ML
ml_client = MLClient(DefaultAzureCredential(), subscription_id="your-subscription-id", resource_group_name="your-resource-group", workspace_name="your-workspace-name")

# Load Titanic dataset from scikit-learn
titanic = fetch_openml(name='titanic', version=1, as_frame=True)
df = titanic.frame

# Save DataFrame to CSV
csv_path = "titanic_dataset.csv"
df.to_csv(csv_path, index=False)

# Get the default datastore
default_datastore = ml_client.datastores.get_default()

# Upload the CSV file to the datastore
datastore_path = f"datasets/titanic/{csv_path}"
data_asset = Data(
    path=csv_path,
    type=AssetTypes.URI_FILE,
    description="Titanic dataset",
    name="titanic_dataset",
    version="1"
)

# Register the dataset
registered_data = ml_client.data.create_or_update(data_asset)

print("Dataset registered. Here is its ID:", registered_data.id)
