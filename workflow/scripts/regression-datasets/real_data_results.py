import sys
import os
sys.path.append(os.getcwd())
import pickle

import numpy as np 
import pandas as pd 

from sklearn.base import clone
from lib import rng_initialisation
from lib import util
from lib import experiment_config as ec

if __name__ == "__main__":
    repeat_ind = int(snakemake.wildcards.repeat_ind)
    dataset = str(snakemake.wildcards.dataset)
    target_name = ec.target_names[dataset]

    def result_record(model_name, mse):
        return {
            "repeat_ind": repeat_ind,
            "model": model_name,
            "mse": mse,
            "dataset": dataset,
        }

    seed = rng_initialisation.get_seed(repeat_ind, dataset, "real-data-learning")
    np.random.seed(seed)

    with open(str(snakemake.input.orig_cols), "rb") as file:
        orig_cols = pickle.load(file)

    train_df = pd.read_csv(str(snakemake.input.train), index_col=False)
    train_oh_df = util.get_oh_df(train_df, target_name, orig_cols)
    X_train, y_train = util.get_X_y(train_oh_df, target_name)

    test_df = pd.read_csv(str(snakemake.input.test), index_col=False)
    test_oh_df = util.get_oh_df(test_df, target_name, orig_cols)
    X_test, y_test = util.get_X_y(test_oh_df, target_name)


    algos = ec.regression_algos

    records = []
    for algo_name, model in algos.items():
        model = clone(model)
        model = model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = np.mean((predictions - y_test)**2)
        records.append(result_record(algo_name, mse))

    result_df = pd.DataFrame.from_records(records)
    result_df.to_csv(str(snakemake.output), index=False)
