import os 
from sklearn.preprocessing import LabelEncoder
import nltk
import pandas as pd
import logging
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')

# creating a log directory to handle all logs
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# configuring logger
logger = logging.getLogger("preprocessing")
logger.setLevel("DEBUG")

# creating a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")


log_file_path = os.path.join(log_dir,'preprocessing.log')
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

def transform_text(text):
    """ transforming the text by converting into lowercase fist then tokenizing them then removing stop words and applying stemming """
    ps = PorterStemmer()
    text = text.lower() # converting to lowercase
    text = nltk.word_tokenize(text) # tokenizing the words
    text = [word for word in text if word.isalnum()] # removing non-alphanumeric tokens
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation] # removing stopwords and punctuation
    text = [ps.stem(word) for word in text] # stemming the words
    return " ".join(text) # joinging the tokens 

def preprocess_df(df,text_col = 'text',target_col = 'target'):
    """ Encoding the target column and transforming the text column """
    try:
        logger.debug("Starting the preprocessing")
        encoder = LabelEncoder()
        df[target_col] = encoder.fit_transform(df[target_col])
        logger.debug("Encoded Target column")

        df = df.drop_duplicates()
        logger.debug("Removed Duplicates")

        df.loc[:, text_col] = df[text_col].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise


def main(text_column='text', target_column='target'):
    try:
        train_data = pd.read_csv('./data/raw/train_data.csv')
        test_data = pd.read_csv('./data/raw/test_data.csv')
        logger.debug('Data loaded')

        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()