import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.logger.logging import logger

class DataPreprocessing:

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def data_read(self,path) -> pd.DataFrame:
        logger.info("The data pre-processing stage has started.")
        try:
            data = pd.read_csv(path)
            logger.info('The file has been read successfully')

            # Date → Season
            data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
            data['months'] = data['Date'].dt.month_name()

            data.loc[data['months'].isin(['January','February','March']), 'season'] = 'Winter'
            data.loc[data['months'].isin(['April','May','June']), 'season'] = 'Autumn'
            data.loc[data['months'].isin(['July','August','September']), 'season'] = 'Summer'
            data.loc[data['months'].isin(['October','November','December']), 'season'] = 'Spring'

            data.drop(columns=['Date','months'], inplace=True)
            data['season'] = self.encoder.fit_transform(data['season'])

            # Rename target column for consistency
            data.rename(columns={'Weekly_Sales':'target'}, inplace=True)

            logger.info('Feature engineering & encoding completed.')
            return data

        except FileNotFoundError:
            logger.error('File not found. Check file path.')
            raise

    def remove_outliers_iqr(self, df: pd.DataFrame, multiplier: int = 2) -> pd.DataFrame:
        logger.info('outliers removal stage')
        mask = pd.Series(True, index=df.index)
        for col in ['Temperature','Fuel_Price','CPI','Unemployment']:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - multiplier * IQR, Q3 + multiplier * IQR
            mask &= (df[col] >= lower) & (df[col] <= upper)

        df_clean = df.loc[mask].reset_index(drop=True)
        logger.info(f"Outliers removed. Rows reduced from {len(df)} to {len(df_clean)}.")
        logger.info('outliers has been removed')
        return df_clean

    def scale(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        logger.info('scaing has been performed')
        numeric = df_clean[['Temperature','Fuel_Price','CPI','Unemployment']]
        scaled = pd.DataFrame(self.scaler.fit_transform(numeric), columns=numeric.columns)

        # add back non-scaled columns
        scaled['Store'] = df_clean['Store'].values
        scaled['season'] = df_clean['season'].values
        scaled['Holiday_Flag'] = df_clean['Holiday_Flag'].values
        scaled['target'] = df_clean['target'].values

        logger.info("Scaling completed.")
        return scaled

    def save_file(self, df_clean: pd.DataFrame,l):
        logger.info('file creation stage')
        path = os.path.join('/Users/gautammehta/Desktop/walmart_sales_forecast_project/data','post_pro')
        os.makedirs(path, exist_ok=True)
        df_clean.to_csv(os.path.join(path, l), index=False)
        logger.info("File saved successfully. Stage has been marked completed.")

if __name__=='__main__':
    #train data
    obj = DataPreprocessing()
    path='/Users/gautammehta/Desktop/walmart_sales_forecast_project/data/raw_data/trained/train_data.csv'
    df = obj.data_read(path)
    df = obj.remove_outliers_iqr(df)
    df = obj.scale(df)
    obj.save_file(df,l='post_pro_train.csv')

    # test data
    obj1 = DataPreprocessing()
    path1='/Users/gautammehta/Desktop/walmart_sales_forecast_project/data/raw_data/test/test_data.csv'
    df1 = obj1.data_read(path1)
    df1 = obj1.remove_outliers_iqr(df1)
    df1 = obj1.scale(df1)
    obj.save_file(df1,l='post_pro_test.csv')