import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.logger.logging import logger

class DataPreprocessing:

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def data_read(self) -> pd.DataFrame:
        logger.info("The data pre-processing stage has started.")
        try:
            data = pd.read_csv('/Users/gautammehta/Desktop/walmart_sales_forecast_project/data/raw_data/trained/train_data.csv')
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
        mask = pd.Series(True, index=df.index)
        for col in ['Temperature','Fuel_Price','CPI','Unemployment']:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - multiplier * IQR, Q3 + multiplier * IQR
            mask &= (df[col] >= lower) & (df[col] <= upper)

        df_clean = df.loc[mask].reset_index(drop=True)
        logger.info(f"Outliers removed. Rows reduced from {len(df)} to {len(df_clean)}.")
        return df_clean

    def scale(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        numeric = df_clean[['Temperature','Fuel_Price','CPI','Unemployment']]
        scaled = pd.DataFrame(self.scaler.fit_transform(numeric), columns=numeric.columns)

        # add back non-scaled columns
        scaled['Store'] = df_clean['Store'].values
        scaled['season'] = df_clean['season'].values
        scaled['Holiday_Flag'] = df_clean['Holiday_Flag'].values
        scaled['target'] = df_clean['target'].values

        logger.info("Scaling completed.")
        return scaled

    def save_file(self, df_clean: pd.DataFrame):
        path = os.path.join('/Users/gautammehta/Desktop/walmart_sales_forecast_project/data','post_pro')
        os.makedirs(path, exist_ok=True)
        df_clean.to_csv(os.path.join(path, 'processed_data_train.csv'), index=False)
        logger.info("Processed data saved successfully.")

if __name__=='__main__':
    obj = DataPreprocessing()
    df = obj.data_read()
    df = obj.remove_outliers_iqr(df)
    df = obj.scale(df)
    obj.save_file(df)
