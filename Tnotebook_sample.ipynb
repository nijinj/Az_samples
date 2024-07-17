# In a Jupyter Notebook

# 1. Connect to Azure ML
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# 2. Load data
import pandas as pd
titanic_data = ml_client.data.get("titanic_dataset")
df = pd.read_csv(titanic_data.path)

# 3. Explore data
print(df.head())
print(df.describe())

# 4. Prepare data
X = df.drop(['Survived'], axis=1)
y = df['Survived']

# 5. Set up AutoML job
from azure.ai.ml.automl import AutoMLClassification

automl_job = AutoMLClassification(
    compute="your-compute-name",
    training_data=df,
    target_column_name="Survived",
    primary_metric="accuracy",
    experiment_name="titanic-experiment"
)

# 6. Run the job
returned_job = ml_client.jobs.create_or_update(automl_job)
ml_client.jobs.stream(returned_job.name)

# 7. Get results
best_model = ml_client.jobs.get(returned_job.name).outputs.best_model
print(f"Best model: {best_model}")
