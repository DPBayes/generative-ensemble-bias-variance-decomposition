import sys
import os
sys.path.append(os.getcwd())
import pickle

import numpy as np 
import pandas as pd 

from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, roc_auc_score
from lib import rng_initialisation
from lib import util
from lib import experiment_config as ec

if __name__ == "__main__":
    repeat_ind = int(snakemake.wildcards.repeat_ind)
    dataset = str(snakemake.wildcards.dataset)
    target_name = ec.target_names[dataset]

    def result_record(model_name, brier, log_loss, accuracy, auc):
        return {
            "repeat_ind": repeat_ind,
            "model": model_name,
            "brier": brier,
            "log_loss": log_loss,
            "accuracy": accuracy,
            "auc": auc,
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


    algos = ec.classification_algos

    records = []
    for algo_name, model in algos.items():
        model = clone(model)
        model = model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)

        brier = brier_score_loss(y_test, preds[:,1])
        log_loss_val = log_loss(y_test, preds)
        accuracy = accuracy_score(y_test, preds[:,1] > 0.5)
        auc = roc_auc_score(y_test, preds[:,1])
        records.append(result_record(algo_name, brier, log_loss_val, accuracy, auc))

    result_df = pd.DataFrame.from_records(records)
    result_df.to_csv(str(snakemake.output), index=False)
