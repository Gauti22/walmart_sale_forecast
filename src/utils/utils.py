import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger.logging import logger
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        logger.fatal('function save_object is not executed.')

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,y_train)

            y_test_pred=model.predict(X_test)

            test_model_score_r2=r2_score(y_test,y_test_pred)
            mse=mean_absolute_error(y_test,y_test_pred)

            report[list(models.keys())[i]]=[f'r2_score: {test_model_score_r2} mse: {mse}']
        
        return report
    except Exception as e:
        logger.fatal('function evaluation is not executed.')
    

def load_object(file_path):
    try:
        print(f"📂 Attempting to load object from: {file_path}")
        with open(file_path, 'rb') as f:
            obj=pickle.load(f)
        print(f" Object loaded successfully from: {file_path}")
        return obj
    except Exception as e:
        logger.info('Exception occured in load_object function utils')
        
    