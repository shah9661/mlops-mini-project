import numpy as np
import pandas as pd
import logging
import yaml
from sklearn.linear_model import LogisticRegression
import pickle
import os

logger=logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
file_handler=logging.FileHandler('model_building.log')

formater=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formater)
file_handler.setFormatter(formater)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_prams(config_path:str)->dict:
    try:
        with open(config_path) as file:
            config=yaml.safe_load(file)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading configuration: {e}")
        raise

def load_data(path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(path,keep_default_na=False)
        logger.info(f"Data loaded successfully from {path}. Data shape: {df.shape}")
        return df
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Data file is empty: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing data file: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")
        raise


def train_model(X_train_df:np.ndarray,y_train:np.ndarray,c,penalty,solver)->LogisticRegression:
    try:
        clf=LogisticRegression(C=c,penalty=penalty,solver=solver)
        clf.fit(X_train_df,y_train)
        logger.debug('Model train complete')
        return clf
    except Exception as e:
        logger.error(f'Error during train model {e}')
        raise

def save_model(model,file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug(f'Model saved {file_path}')

    except KeyboardInterrupt as e:
        logger.error(f'KeyboardInterrput {e}')
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading configuration: {e}")
        raise

def main():
    try:
        config=load_prams('params.yaml')
        c=config['model_building']['C']
        penalty=config['model_building']['penalty']
        solver=config['model_building']['solver']

        train=load_data("./data/processed/train_bow.csv")
        X_train=train.iloc[:,0:-1].values
        y_train=train.iloc[:,-1].values
        model=train_model(X_train,y_train,c,penalty,solver)
        save_model(model,'models/model.pkl')
    except Exception as e:
        logger.error(f'Failed to complete the model {e}')
        raise e

if __name__=="__main__":
    main()