import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
from src.logger.logging import logger
from src.utils.utils import evaluate_model
import json
import os

def model_creation():
    data=pd.read_csv('/Users/gautammehta/Desktop/walmart_sales_forecast_project/data/post_pro/post_pro_train.csv')
    data1=pd.read_csv('/Users/gautammehta/Desktop/walmart_sales_forecast_project/data/post_pro/post_pro_test.csv')
    X_train=data.iloc[:,:7]
    y_train=data.iloc[:,7:]
    X_test=data1.iloc[:,:7]
    y_test=data1.iloc[:,7:]

    logger.info('Model building has been started')

    models={
    'LinearRegression':LinearRegression(),
    'Ridge':Ridge(),
    'KNeighborsRegressor':KNeighborsRegressor(),
    'RandomForestRegressor':RandomForestRegressor(),
    'DecisionTreeRegressor':DecisionTreeRegressor(),
    'XGBRegressor':XGBRegressor()}

    evaluate_model(X_train,X_test,y_test,y_train,models)
    logger.info('model has been created')

    model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
    path=os.path.join('models','reports')
    os.makedirs(path,exist_ok=True)
    file_path=os.path.join(path,'model_report.json')
    with open(file_path,'w') as f:
        json.dump(model_report,f,indent=4)
        logger.info('the report has been saved in report folder')

    best_model_score=max(sorted(model_report.values()))

    best_model_name = list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
    ]

    best_model = models[best_model_name]

    file_path=os.path.join(path,'best_model.json')
    with open(file_path,'w') as f:
        json.dump(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}',f,indent=4)
        logger.info('best model metric has been saved')
    path = os.path.join("models", "saved_models")
    os.makedirs(path, exist_ok=True)

    # File path
    file_path = os.path.join(path, "best_model.pkl")

    # Save the model
    with open(file_path, "wb") as f:   # 'wb' = write binary
        pickle.dump(best_model, f)
        logger.info('model has been saved in pkl')


if __name__=='__main__':
    model_creation()