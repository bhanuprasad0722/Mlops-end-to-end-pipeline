import pandas as pd 
import os
import logging
from sklearn.model_selection import train_test_split

# creating a log directory to handle all logs
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# configuring logger
logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

# creating a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")


log_file_path = os.path.join(log_dir,'data_ingestion.log')
# creating a file handler
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

# creating a formatter 
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# setting the formatter to both console and file handler's
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# adding handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url):
    """ takes data url as an input and loads data into a df """
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s",data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the file: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured: %s",e)
        raise

def preprocess_data(df):
    """ Basic preprocessing """
    try:
        df.drop(columns = ['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace = True)
        df.rename(columns = {'v1':'target','v2':'text'},inplace = True)
        logger.debug("Data preprocessing Completed")
        return df
    except KeyError as e:
        logger.error("missing column in the dataframe %s",e)
        raise
    except Exception as e:
        logger.error("Unknow error occured while preprocessing the data %s",e)
        raise

def save_data(test_data,train_data,data_path):
    """ Saving the data """
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        test_data.to_csv(os.path.join(raw_data_path,'test_data.csv'),index = False)
        train_data.to_csv(os.path.join(raw_data_path,'train_data.csv'),index = False)
        logger.debug("Train and Test data saved to path %s",raw_data_path)
    except Exception as e:
        logger.error("Unexpected Error Occured %s",e)
        raise

def main():
    try:
        data_url = 'https://raw.githubusercontent.com/bhanuprasad0722/Mlops-end-to-end-pipeline/refs/heads/main/experiments/spam.csv'
        df = load_data(data_url)
        processed_df = preprocess_data(df)
        train_data,test_data = train_test_split(processed_df,test_size=20,random_state=42)
        save_data(train_data,test_data,data_path = './data')
    except Exception as e:
        logger.error("Failed to complete data ingestion process %s",e)
        raise

if __name__ == '__main__':
    main()



