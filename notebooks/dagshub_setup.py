import mlflow
import dagshub

dagshub.init(repo_owner='shanshad0999', repo_name='mlops-mini-project', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/shanshad0999/mlops-mini-project.mlflow')

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)