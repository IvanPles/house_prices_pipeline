from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                             mean_absolute_percentage_error, r2_score)
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import json
import mlflow
import numpy as np
import os

SCALERS = {
    'robust': RobustScaler,
    'standard': StandardScaler,
    'minmax': MinMaxScaler
}

MODELS = {
    'ridge': Ridge,
    'gradient_boosting': GradientBoostingRegressor,
    'random_forest': RandomForestRegressor
}

def calculate_metrics(y_true, y_pred, prefix=''):
    res = {f'{prefix}mae': mean_absolute_error(y_true, y_pred),
          f'{prefix}rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
          f'{prefix}rmsle': np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))),
          f'{prefix}mape': mean_absolute_percentage_error(y_true, y_pred)*100,
          f'{prefix}r2': r2_score(y_true, y_pred)}
    return res

def create_plot(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.plot(y_true, y_pred, 's')
    ax.plot(np.linspace(0, max(y_true)), np.linspace(0, max(y_true)))
    ax.legend(['predicted', 'y=x'])
    ax.grid()
    ax.set_ylabel('predicted cost')
    ax.set_ylabel('real cost') 
    return fig, ax


if __name__ == '__main__':
    data_train, data_test = pd.read_csv('data/train_prepped.csv'), pd.read_csv('data/test_prepped.csv')
    inps = [c for c in data_train.columns if 'SalePrice' not in c ]
    y_train_log, y_test_log = data_train['SalePrice_log'], data_test['SalePrice_log']
    y_train, y_test = data_train['SalePrice'], data_test['SalePrice']
    X_train, X_test = data_train[inps], data_test[inps]
    with open('models_config.json', 'r') as f:
        models_config = json.load(f)
    os.makedirs("models", exist_ok=True)
    for name, m in models_config.items():
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(name)
        with mlflow.start_run():
            curr_model = MODELS[m['type']](**m['params'])
            scaler = SCALERS.get(m['scaler'], None)
            mlflow.log_params(m['params'])
            mlflow.log_param('scaler', m['scaler'])
            if scaler is not None:
                pipeline = make_pipeline(scaler(), curr_model)
            else:
                pipeline = curr_model
            print(f'fitting {name}')
            pipeline.fit(X_train, y_train_log)
            mlflow.sklearn.log_model(pipeline, f'model/{name}')
            ###
            pred_train = pipeline.predict(X_train)
            pred_train = np.exp(pred_train)-1
            train_metrics = calculate_metrics(y_train, pred_train, 'train_')
            mlflow.log_metrics(train_metrics)
            ###
            pred_test = pipeline.predict(X_test)
            pred_test = np.exp(pred_test)-1
            test_metrics = calculate_metrics(y_test, pred_test, 'test_')
            mlflow.log_metrics(test_metrics)
            ### 
            fig, ax = create_plot(y_test, pred_test)
            mlflow.log_figure(fig, f'figs/{name}.png')
            ###
            with open(f'models/{name}.pkl', 'wb') as f:
                pickle.dump(pipeline, f)
        