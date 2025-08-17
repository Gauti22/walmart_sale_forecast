import pandas as pd
import numpy as np
import os
from src.logger.logging import logger
from sklearn.model_selection import train_test_split



class DataIngestionConfig():

    def __init__(self):
        self.raw_data=os.path.join('File_raw','raw_data')
        self.train_data=os.path.join('data/raw_data','trained')
        self.test_data=os.path.join('data/raw_data','test')

class DataIngestion():
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        self.logger=logger

    def initiate_data_ingestion(self):
        self.logger.info('the data ingestion process has been started')
        try:
            data=pd.read_csv('/Users/gautammehta/Downloads/Walmart.csv')
            self.logger.info('the dataframe has been created from the raw file.')
            os.makedirs(os.path.join(self.ingestion_config.raw_data),exist_ok=False)
            data.to_csv(os.path.join(self.ingestion_config.raw_data,'raw_data.csv'))
            self.logger.info('data has been saved to File_raw folder')

            self.logger.info('performing train_test split')
            train_data,test_data=train_test_split(data,test_size=0.20)

            self.logger.info('the train_test has been perfomred from the raw file.')
            os.makedirs(os.path.join(self.ingestion_config.train_data),exist_ok=False)
            train_data.to_csv(os.path.join(self.ingestion_config.train_data,'train_data.csv'))
            self.logger.info('data has been saved to train_data')

            self.logger.info('the test_data has been perfomred from the raw file.')
            os.makedirs(os.path.join(self.ingestion_config.test_data),exist_ok=False)
            test_data.to_csv(os.path.join(self.ingestion_config.test_data,'test_data.csv'))
            self.logger.info('data has been saved to test_data')

            self.logger.info("Train and test data saved")

            return(

                self.ingestion_config.train_data,
                self.ingestion_config.test_data,
               
            )
        except FileNotFoundError :
            self.logger.fatal('path provided has no file please check')
        except ValueError :
            self.logger.fatal('wrong value has been provided')
        except FileExistsError :
            self.logger.fatal('file not found. entire code will not work')


if __name__=='__main__':
    obj=DataIngestion()
    obj.initiate_data_ingestion()
