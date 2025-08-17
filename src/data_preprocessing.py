import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


logger=logging.getLogger('data_preprocessing')
logger.setLevel(logging.INFO)
log_path = os.path.join("logs", "app.log")  # logs/app.log
os.makedirs("logs", exist_ok=True)          # Create folder if it doesn't exist
handler = logging.FileHandler(log_path, mode='a')
fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
logger.addHandler(handler)


def data_read(path_x:str,path_y:str)->pd.DataFrame:
    try:
        df_X=pd.read_csv('/Users/gautammehta/Desktop/walmart_sales_forecast_project/data/raw/X_train_raw.csv')
        df_y=pd.read_csv('/Users/gautammehta/Desktop/walmart_sales_forecast_project/data/raw/y_train_raw.csv')
        df=df_X.copy()
        df['target']=df_y['Weekly_Sales']
        logger.info('the file has been processed successfully')
        return(df)
    except FileExistsError:
        logger.fatal('file no found')
    except FileNotFoundError:
        logger.warning('check file path script is not executed')
    

# date time data type and converting the dates to season.
def date_time(df:pd.DataFrame)->pd.DataFrame:
    # change date to date_time
    df['Date']=pd.to_datetime(df['Date'],dayfirst=True)
    df['months']=df['Date'].dt.month_name()
    df.drop(columns='Date',inplace=True)
    df['season']=''
    for x in range(len(df['months'])):
        if df['months'].iloc[x] in ['January','Feburary','March']:
            df['season'].iloc[x]='Winter'
        elif df['months'].iloc[x] in ['April','May','June']:
            df['season'].iloc[x]='Aut'
        elif df['months'].iloc[x] in ['July','Aug','Sept']:
            df['season'].iloc[x]='Summer'
        else:
            df['season'].iloc[x]='Spring'
    df.drop(columns='months',inplace=True)
    return(df)

# encoding
def encoding(df:pd.DataFrame)->pd.DataFrame:
    le=LabelEncoder()
    df['season']=le.fit_transform(df['season'])
    return(df)

# removing outliers
def remove_outliers_iqr(df:pd.DataFrame, multiplier:int=2)->pd.DataFrame:
    df_clean = df.copy()
    h=df['Holiday_Flag']
    y=df['target']
    df_clean.drop(columns=['Holiday_Flag','target'],inplace=True)
    for col in df_clean.select_dtypes(include='number').columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Keep only rows within bounds
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    df_clean['Holiday_Flag']=h
    df_clean['target']=y
    return df_clean

# scaling
def scale(df_clean:pd.DataFrame)->pd.DataFrame:
    scaling=StandardScaler()
    y=df_clean['target']
    df_clean.drop(columns=['target'],inplace=True)
    df_clean=pd.DataFrame(scaling.fit_transform(df_clean),columns=df_clean.columns)
    df_clean['target']=y
    return(df_clean)


def file(df_clean:pd.DataFrame):
    path=os.path.join('/Users/gautammehta/Desktop/walmart_sales_forecast_project/data','processed')
    os.makedirs(path,exist_ok=True)
    df_clean.to_csv(os.path.join(path,'processed_data'))

def main():
    df=data_read('/Users/gautammehta/Desktop/walmart_sales_forecast_project/data/raw/X_train_raw.csv','/Users/gautammehta/Desktop/walmart_sales_forecast_project/data/raw/y_train_raw.csv')
    df=date_time(df)
    df=encoding(df)
    df=remove_outliers_iqr(df)
    df=scale(df)
    df=file(df)

if __name__=='__main__':
    main()