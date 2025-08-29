import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json


def fill_cols(df, fill_map):
    for c, val in fill_map.items():
        df[c] = df[c].fillna(val)
    return df

def turn_to_categorical(df, to_categorical):
    for c in to_categorical:
        df[c] = df[c].astype(str)
    return df

def cut_outliers(df, bounds_map):
    for c, bounds in bounds_map.items():
        min_v, max_v = bounds
        df = df[(df[c]>min_v)&(df[c]<max_v)]
    return df

def featurize(df):
    df['haspool'] = (df['PoolArea']>0.).astype(int)
    df['has2ndfloor'] = (df['2ndFlrSF']>0.).astype(int)
    df['hasgarage'] = (df['GarageArea']>0.).astype(int)
    df['hasbsmt'] = (df['TotalBsmtSF']>0.).astype(int)
    df['hasfireplace'] = (df['Fireplaces']>0.).astype(int)
    return df

def process_categorical(df):
    numerical = df.select_dtypes(include=['int', 'float'])
    dfs_cat = []
    for c in df.select_dtypes(exclude=['int', 'float']).columns:
        df_cat = pd.get_dummies(df[c], prefix=c)
        dfs_cat.append(df_cat.loc[:, df_cat.columns[0:-1]])
    categorical = pd.concat(dfs_cat, axis=1)
    return pd.concat([numerical, categorical], axis=1)


if __name__ == '__main__':
    data = pd.read_csv('data/train.csv')
    with open('data_process_config.json', 'r') as f:
        config = json.load(f)
    # remove nans
    if config['log_norm']:
        data['SalePrice_log'] = np.log1p(data["SalePrice"])
    data = fill_cols(data, config['fill_na'])
    data = turn_to_categorical(data, config['to_categorical'])
    data = cut_outliers(data, config['outliers'])
    data = featurize(data)
    data = data.drop(columns=config['cols_to_drop'])
    data = process_categorical(data)

    data_train, data_test = train_test_split(data, test_size=config.get('test_size', 0.1), random_state=42)
    data_train.to_csv('data/train_prepped.csv', index=False)
    data_test.to_csv('data/test_prepped.csv', index=False)


