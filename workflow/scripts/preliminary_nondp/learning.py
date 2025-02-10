import sys
import os
sys.path.append(os.getcwd())
import pickle

import numpy as np 
import pandas as pd 
import torch
torch.set_default_dtype(torch.float64)

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from lib import rng_initialisation
from lib import experiment_config as ec


if __name__ == "__main__":
    repeat_ind = int(snakemake.wildcards.repeat_ind)
    method = str(snakemake.wildcards.method)

    def result_record(model_name, mse, n_syn_datasets):
        return {
            "repeat_ind": repeat_ind,
            "model": model_name,
            "mse": mse,
            "n_syn_datasets": n_syn_datasets,
            "method": method,
        }

    seed = rng_initialisation.get_seed(repeat_ind, method, "california-housing", "learning")
    np.random.seed(seed)

    algos = ec.regression_algos

    test_df = pd.read_csv(str(snakemake.input.test), index_col=False)
    X_test = test_df.values[:, :-1]
    y_test = test_df.values[:, -1]

    syn_dfs = []
    for filename in snakemake.input.syn_data:
        syn_dfs.append(pd.read_csv(str(filename), index_col=False))

    trained_models = {
        algo_name: list() for algo_name in algos.keys()
    }
    for syn_df in syn_dfs:
        X_train_syn = syn_df.values[:, :-1]
        y_train_syn = syn_df.values[:, -1]
        for name, model in algos.items():
            model = clone(model)
            model = model.fit(X_train_syn, y_train_syn)
            trained_models[name].append(model)

    n_syn_datasets_values = [1, 2, 5, 10]
    # n_syn_datasets_values = [1, 2]

    result_records = []
    for n_syn_datasets in n_syn_datasets_values:
        for algo_name, models in trained_models.items():
            selected_models = models[:n_syn_datasets]
            predictions = np.zeros((y_test.shape[0], n_syn_datasets))
            for i in range(n_syn_datasets):
                predictions[:, i] = selected_models[i].predict(X_test)
            ensemble_predictions = predictions.mean(axis=1)
            ensemble_mse = np.mean((ensemble_predictions - y_test)**2)
            result_records.append(result_record(algo_name, ensemble_mse, n_syn_datasets))

    result_df = pd.DataFrame.from_records(result_records)
    result_df.to_csv(str(snakemake.output), index=False)