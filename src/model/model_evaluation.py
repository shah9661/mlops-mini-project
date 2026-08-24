import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import json
import mlflow
import mlflow.sklearn
import dagshub
import os


dagshub.init(repo_owner='shanshad0999', repo_name='mlops-mini-project', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/shanshad0999/mlops-mini-project.mlflow')

logger=logging.getLogger("Model evaluation")
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
file_handler=logging.FileHandler('model_evaluation.log')

formater=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formater)
file_handler.setFormatter(formater)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(model_path:str):
    try:
        with open(model_path,'rb') as file:
            model=pickle.load(file)
        logger.debug(f'Model loaded {model_path}')
        return model
    except FileNotFoundError as e:
        logger.error(f'File not found error {model_path}')
        raise
    except Exception as e:
        logger.error(f'Unwanted error {e}')
        raise

def load_test_data(file_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        logger.debug(f"Test file loaded {file_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f'file to parase csv file {e}')
        raise
    except Exception as e:
        logger.error(f"Unwanted error {e}")
        raise

def evaluation(clf,X_test:np.ndarray,y_test:np.ndarray)->dict:
    try:
        y_pred = clf.predict(X_test)
        logger.debug(f'Model predicted')
        y_pred_proba = clf.predict_proba(X_test)
        logger.debug(f'Model predicted proba')
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error("Error during model evaluation")
        raise

    
def save_metrics(metrix_dict:dict,file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,'w') as file:
            json.dump(metrix_dict,file,indent=4)
        logger.debug(f"Metrics saved {file_path}")
    except Exception as e:
        logger.error(f'Error occurred while saving the metrics: ,{e}')
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    mlflow.set_experiment("DVC pipeline")
    with mlflow.start_run() as run:
        try:
            model=load_model("./models/model.pkl")
            df=load_test_data("./data/processed/test_bow.csv")
            X_test=df.iloc[:,0:-1].values
            y_test=df.iloc[:,-1].values
            metrics=evaluation(model,X_test,y_test)
            save_metrics(metrics,"./reports/metrics.json")

            for metric_name, metric_value in metrics.items():
                 mlflow.log_metric(metric_name, metric_value)
            if hasattr(model, 'get_params'):
                params = model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            logged_model = mlflow.sklearn.log_model(model,name="model")
            
            save_model_info(run.info.run_id, logged_model.model_uri, 'reports/experiment_info.json')
            mlflow.log_artifact('reports/metrics.json')

            mlflow.log_artifact("reports/experiment_info.json")

            mlflow.log_artifact('model_evaluation.log')

        except Exception as e:
            logger.error(f"Unwanted error {e}")

if __name__=="__main__":
    main()