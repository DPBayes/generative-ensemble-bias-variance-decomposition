import sys
import os
sys.path.append(os.getcwd())
import pickle

import numpy as np 
import pandas as pd 

from sklearn.base import clone
from lib import rng_initialisation
from lib import experiment_config as ec

if __name__ == "__main__":
    repeat_ind = int(snakemake.wildcards.repeat_ind)

    def result_record(model_name, mse):
        return {
            "repeat_ind": repeat_ind,
            "model": model_name,
            "mse": mse,
        }

    seed = rng_initialisation.get_seed(repeat_ind, "california-housing", "real-data-learning")
    np.random.seed(seed)

    train_df = pd.read_csv(str(snakemake.input.train), index_col=False)
    X_train = train_df.values[:, :-1]
    y_train = train_df.values[:, -1]

    test_df = pd.read_csv(str(snakemake.input.test), index_col=False)
    X_test = test_df.values[:, :-1]
    y_test = test_df.values[:, -1]

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
