import pandas as pd
import logging
import yaml
from sklearn.feature_extraction.text import CountVectorizer
import os
import pickle

logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('feature_engineering_errors.log')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

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

def apply_bow(train_df:pd.DataFrame,test_df:pd.DataFrame,max_features:int)->tuple:
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        x_train=train_df['content'].values
        y_train=train_df['sentiment'].values

        x_test=test_df['content'].values
        y_test=test_df['sentiment'].values
        x_train_bow=vectorizer.fit_transform(x_train)
        x_test_bow=vectorizer.transform(x_test)
        train_df=pd.DataFrame(x_train_bow.toarray())
        train_df['label']=y_train
        test_data_df=pd.DataFrame(x_test_bow.toarray())
        test_data_df['label']=y_test
        pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
        
        logger.info(f"bag of Words transformation applied successfully. Train shape: {train_df.shape}, Test shape: {test_data_df.shape}")
        return train_df, test_data_df
    except KeyError as e:
        logger.error(f"Key error occurred during bag of Words transformation: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while applying Bag of Words transformation: {e}")
        raise
def save_data(df:pd.DataFrame,path:str,filename:str)->None:
    try:
        os.makedirs(path, exist_ok=True)
        df.to_csv(os.path.join(path,filename), index=False)
        logger.info(f"Data saved successfully to {os.path.join(path,filename)}")
    except FileNotFoundError as e:
        logger.error(f"File not found error occurred while saving data: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving data: {e}")
        raise
def main():
    try:
        config=load_prams('params.yaml')
        max_features=config['feature_engineering']['max_features']
        train_df=load_data('./data/interim/train_processed.csv')
        test_df=load_data('./data/interim/test_processed.csv')
        train_bow,test_bow=apply_bow(train_df,test_df,max_features)
        save_data(train_bow,'./data/processed','train_bow.csv')
        save_data(test_bow,'./data/processed','test_bow.csv')
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main function: {e}")
        raise

if __name__=='__main__':
    main()