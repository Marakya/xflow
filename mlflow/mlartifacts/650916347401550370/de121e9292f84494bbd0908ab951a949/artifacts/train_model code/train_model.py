from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd
import os
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_model")

df = pd.read_csv('/home/xflow/project/datasets/data_train.csv', header=None)
df.columns = ['id', 'counts']
df['counts'] = df['counts'].fillna(0)

model = RandomForestRegressor(max_depth=2, random_state=0)

with mlflow.start_run():
    mlflow.sklearn.log_model(model,
                             artifact_path="lr",
                             registered_model_name="lr")
    mlflow.log_artifact(local_path="/home/xflow/project/scripts/train_model.py",
                        artifact_path="train_model code")
    mlflow.end_run()

model.fit(df['id'].values.reshape(-1,1), df['counts'])

with open('/home/xflow/project/models/data.pickle', 'wb') as f:
    pickle.dump(model, f)
