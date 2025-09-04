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
formatter = logging.Formatter("%(asctimme)s - %(name)s - %(levelname)s - %(message)s")

# setting the formatter to both console and file handler's
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# adding handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)



