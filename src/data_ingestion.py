import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split

logger=logging.getLogger('my_app')
logger.setLevel(logging.INFO)
log_path = os.path.join("logs", "app.log")  # logs/app.log
os.makedirs("logs", exist_ok=True)          # Create folder if it doesn't exist
handler = logging.FileHandler(log_path, mode='a')
fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
logger.addHandler(handler)


def read_data(path):
    try:
        df=pd.read_csv(path)
        logger.info('the dataframe has been created from the raw file.')
        return(df)
    except FileNotFoundError :
        logger.fatal('path provided has no file please check')
    except ValueError :
        logger.fatal('wrong value has been provided')
    except FileExistsError :
        logger.fatal('file not found. entire code will not work')

def split(df):
    y=df['Weekly_Sales']
    df.drop(columns='Weekly_Sales',inplace=True)
    X=df
    try:
        X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,train_size=0.8,random_state=42)
        path=os.path.join('data','raw')
        os.makedirs(path,exist_ok=True)
        X_train.to_csv(os.path.join(path,'X_train_raw.csv'),index=False)
        X_test.to_csv(os.path.join(path,'X_test_raw.csv'),index=False)
        y_train.to_csv(os.path.join(path,'y_train_raw.csv'),index=False)
        y_test.to_csv(os.path.join(path,'y_test_raw.csv'),index=False)
        logger.info('The data has been splitted successfully')
    except KeyError:
        logger.error('there is some issue in the code. Please check')


def main():
    df=read_data('/Users/gautammehta/Desktop/walmart_sales_forecast_project/File_raw/Walmart.csv')
    split(df)

if __name__=='__main__':
    main()
