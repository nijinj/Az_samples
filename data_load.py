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


#Clen if required

import pandas as pd
from sklearn.impute import SimpleImputer

# Example DataFrame 'df' loaded here
# df = pd.read_csv("your_data.csv")

# Create an imputer object for numeric columns using the mean
numeric_imputer = SimpleImputer(strategy='mean')

# For categorical data, you might want to use a constant such as 'missing' or the mode
categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')

# Apply imputers selectively
for column in df.columns:
    if df[column].dtype.kind in 'biufc':  # Numeric columns
        df[column] = numeric_imputer.fit_transform(df[[column]])
    else:  # Categorical columns
        df[column] = categorical_imputer.fit_transform(df[[column]])

