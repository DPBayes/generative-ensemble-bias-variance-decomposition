from tqdm import tqdm
from sklearn.base import clone
import numpy as np
import pandas as pd
import re

def tqdm_choice(iterator, show_progress):
    if show_progress:
        return tqdm(iterator)
    else: 
        return iterator

def train_model_return_mse(model, X_train, y_train, X_test, y_test):
    model = clone(model)
    model = model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = np.mean((preds - y_test)**2)
    return mse

def get_oh_df(df, target_name, orig_cols=None):
    oh_df = pd.get_dummies(df, drop_first=False)
    oh_df = move_target_col_to_end(oh_df, target_name)
    if orig_cols is not None:
        oh_df = oh_df.reindex(columns=orig_cols, fill_value=0)
    return oh_df

def get_X_y(df, target_name):
    X = df.drop(columns=[target_name])
    y = df[target_name]
    return X.values, y.values

def move_target_col_to_end(df, target_name):
    cols = list(df.columns)
    cols.remove(target_name)
    cols.append(target_name)
    return df.reindex(columns=cols)

def get_first_part(df, n_parts):
    n_part = int(df.shape[0] / n_parts)
    return df.iloc[:n_part]

def get_ith_part(df, n_parts, i):
    part_size = int(df.shape[0] / n_parts)
    return df.iloc[i * part_size : (i+1) * part_size]

def get_prefix_subset(df, n_parts, n_parts_to_get):
    part_size = int(df.shape[0] / n_parts)
    return df.iloc[:part_size * n_parts_to_get]

def interval_middle(interval):
    if type(interval) is str:
        e = re.compile(r"\((\d*\.\d*), (\d*.\d*)]")
        m = e.match(interval)
        return (float(m[1]) + float(m[2])) / 2
    else:
        return interval.mid

def ensure_non_0_predictions(preds, min_proba=0.0001):
    for i in range(preds.shape[0]):
        for j in range(preds.shape[2]):
            if preds[i,0,j] < min_proba:
                preds[i,0,j] += min_proba
                preds[i,1,j] -= min_proba
            elif preds[i,1,j] < min_proba:
                preds[i,1,j] += min_proba
                preds[i,0,j] -= min_proba
    return preds