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
    model_name = str(snakemake.wildcards.model)
    dataset = str(snakemake.wildcards.dataset)
    ec.check_dataset_name(dataset)
    is_classification = ec.is_classification(dataset)

    seed = rng_initialisation.get_seed(repeat_ind, method, model_name, dataset, "variance estimation")
    np.random.seed(seed)

    algos = ec.classification_algos if is_classification else ec.regression_algos
    target_name = ec.target_names[dataset]

    with open(str(snakemake.input.orig_cols), "rb") as file:
        orig_cols = pickle.load(file)

    test_df = pd.read_csv(str(snakemake.input.test), index_col=False)
    test_oh_df = util.get_oh_df(test_df, target_name, orig_cols)
    X_test, y_test = util.get_X_y(test_oh_df, target_name)

    n_generators = len(snakemake.input.syn_data)
    # n_generators = 2

    all_predictions = np.zeros((n_generators, ec.n_generated_syn_datasets, y_test.size))
    for generator_ind in range(n_generators):
        syn_df = pd.read_csv(str(snakemake.input.syn_data[generator_ind]), index_col=False)
        syn_oh_df = util.get_oh_df(syn_df, target_name, orig_cols)
        for syn_dataset_ind in range(ec.n_generated_syn_datasets):
            sel_syn_df = util.get_ith_part(syn_oh_df, ec.n_generated_syn_datasets, syn_dataset_ind)
            X_train_syn, y_train_syn = util.get_X_y(sel_syn_df, target_name)

            model = clone(algos[model_name])
            model = model.fit(X_train_syn, y_train_syn)
            if is_classification:
                preds_test = model.predict_proba(X_test)[:,1]
            else:
                preds_test = model.predict(X_test)
            all_predictions[generator_ind, syn_dataset_ind, :] = preds_test


    model_variances = all_predictions.var(axis=1).mean(axis=0)
    synthetic_data_variances = all_predictions.mean(axis=1).var(axis=0)
    output = {
        "model_variances": model_variances,
        "synthetic_data_variances": synthetic_data_variances,
    }

    with open(str(snakemake.output), "wb") as file:
        pickle.dump(output, file)
