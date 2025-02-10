import sys
import os
sys.path.append(os.getcwd())
import pickle

import numpy as np 
import pandas as pd 

from lib import rng_initialisation
from lib import util
from lib import experiment_config as ec

from sklearn.base import clone


if __name__ == "__main__":
    repeat_ind = int(snakemake.wildcards.repeat_ind)
    method = str(snakemake.wildcards.method)
    dataset = str(snakemake.wildcards.dataset)

    def result_record(model_name, mse_test, n_syn_datasets):
        return {
            "repeat_ind": repeat_ind,
            "model": model_name,
            "mse_test": mse_test,
            "n_syn_datasets": n_syn_datasets,
            "method": method,
        }

    seed = rng_initialisation.get_seed(repeat_ind, method, dataset, "learning")
    np.random.seed(seed)

    algos = ec.regression_algos
    target_name = ec.target_names[dataset]

    with open(str(snakemake.input.orig_cols), "rb") as file:
        orig_cols = pickle.load(file)

    test_df = pd.read_csv(str(snakemake.input.test), index_col=False)
    test_oh_df = util.get_oh_df(test_df, target_name, orig_cols)
    X_test, y_test = util.get_X_y(test_oh_df, target_name)

    syn_preds = {algo_name: {kind: list() for kind in ["test"]} for algo_name in algos.keys()}
    for filename in snakemake.input.syn_preds:
        with open(str(filename), "rb") as file:
            input_obj = pickle.load(file)
        for name in syn_preds.keys():
            for kind in syn_preds[name].keys():
                syn_preds[name][kind].append(input_obj[kind][name])

    for name in syn_preds.keys():
        for kind in syn_preds[name].keys():
            syn_preds[name][kind] = np.stack(syn_preds[name][kind], axis=1)

    n_syn_datasets_values = [1, 2, 4, 8, 16, 32]
    # n_syn_datasets_values = [1, 2]

    result_records = []
    for n_syn_datasets in n_syn_datasets_values:
        for algo_name, all_preds in syn_preds.items():
            predictions_test = all_preds["test"][:, :n_syn_datasets]
            ensemble_predictions_test = predictions_test.mean(axis=1)
            ensemble_mse_test = np.mean((ensemble_predictions_test - y_test)**2)
            result_records.append(result_record(algo_name, ensemble_mse_test, n_syn_datasets))

    result_df = pd.DataFrame.from_records(result_records)
    result_df.to_csv(str(snakemake.output), index=False)