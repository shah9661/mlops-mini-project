import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging
import yaml

logger=logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
file_handler=logging.FileHandler('error.log')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

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
        

def ingest_data(url:str)->pd.DataFrame:
    try:
        df=pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        logger.info(f"Data ingestion successful. Data shape: {df.shape}")
        return df
    except KeyError as e:
        logger.error(f"Key error occurred: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while ingesting data: {e}")
        raise
    
    
def clean_data(df:pd.DataFrame)->pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'],inplace=True)
        df.fillna('',inplace=True)
        logger.info(f"Data cleaning successful. Data shape: {df.shape}")
        return df
    except KeyError as e:
        logger.error(f"Key error occurred during data cleaning: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while cleaning data: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise
    

def main():
    try:
        config=load_prams('params.yaml')
        test_size=config['data_ingestion']['test_size']
        df=ingest_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df=clean_data(df)
        train_data, test_data=train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, 'data')
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}")
        raise
if __name__=="__main__":
    main()


