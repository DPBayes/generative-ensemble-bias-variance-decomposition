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
    size_mul = int(snakemake.wildcards.size_mul)

    def result_record(model_name, mse_test):
        return {
            "repeat_ind": repeat_ind,
            "model": model_name,
            "mse_test": mse_test,
            "size_mul": size_mul,
            "method": method,
        }

    seed = rng_initialisation.get_seed(repeat_ind, method, dataset, size_mul, "one large learning")
    np.random.seed(seed)

    algos = ec.regression_algos
    target_name = ec.target_names[dataset]

    with open(str(snakemake.input.orig_cols), "rb") as file:
        orig_cols = pickle.load(file)

    test_df = pd.read_csv(str(snakemake.input.test), index_col=False)
    test_oh_df = util.get_oh_df(test_df, target_name, orig_cols)
    X_test, y_test = util.get_X_y(test_oh_df, target_name)

    syn_df = pd.read_csv(str(snakemake.input.syn_data), index_col=False)
    syn_df = util.get_prefix_subset(syn_df, ec.n_generated_syn_datasets, size_mul)
    syn_oh_df = util.get_oh_df(syn_df, target_name, orig_cols)
    X_train_syn, y_train_syn = util.get_X_y(syn_oh_df, target_name)

    result_records = []
    for name, model in algos.items():
        model = clone(model)
        model = model.fit(X_train_syn, y_train_syn)
        preds_test = model.predict(X_test)
        mse_test = np.mean((preds_test - y_test)**2)
        result_records.append(result_record(name, mse_test))

    result_df = pd.DataFrame.from_records(result_records)
    result_df.to_csv(str(snakemake.output), index=False)

