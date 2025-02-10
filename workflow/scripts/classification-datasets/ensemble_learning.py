import sys
import os
sys.path.append(os.getcwd())
import pickle

import numpy as np 
import pandas as pd 

from lib import rng_initialisation
from lib import util
from lib import experiment_config as ec

from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, roc_auc_score


if __name__ == "__main__":
    repeat_ind = int(snakemake.wildcards.repeat_ind)
    method = str(snakemake.wildcards.method)
    dataset = str(snakemake.wildcards.dataset)

    def result_record(model_name, brier, log_loss, accuracy, auc, n_syn_datasets, primal):
        return {
            "repeat_ind": repeat_ind,
            "model": model_name,
            "n_syn_datasets": n_syn_datasets,
            "method": method,
            "dataset": dataset,
            "primal": primal,
            "brier": brier,
            "log_loss": log_loss,
            "accuracy": accuracy,
            "auc": auc,
        }

    seed = rng_initialisation.get_seed(repeat_ind, method, dataset, "ensemble learning")
    np.random.seed(seed)

    algos = ec.classification_algos
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
            syn_preds[name][kind] = np.stack(syn_preds[name][kind], axis=2)

    n_syn_datasets_values = [1, 2, 4, 8, 16, 32]
    # n_syn_datasets_values = [1, 2]

    result_records = []
    for n_syn_datasets in n_syn_datasets_values:
        for algo_name, all_preds in syn_preds.items():
            predictions = all_preds["test"][:, :, :n_syn_datasets]

            primal_ensemble_predictions = predictions.mean(axis=2)
            primal_brier = brier_score_loss(y_test, primal_ensemble_predictions[:,1])
            primal_log_loss_val = log_loss(y_test, primal_ensemble_predictions)
            primal_accuracy = accuracy_score(y_test, primal_ensemble_predictions[:,1] > 0.5)
            primal_auc = roc_auc_score(y_test, primal_ensemble_predictions[:,1])
            result_records.append(result_record(algo_name, primal_brier, primal_log_loss_val, primal_accuracy, primal_auc, n_syn_datasets, "Primal"))

            unnorm_dual_ensemble_predictions = np.exp(np.log(util.ensure_non_0_predictions(predictions)).mean(axis=2))
            dual_ensemble_predictions = unnorm_dual_ensemble_predictions / unnorm_dual_ensemble_predictions.sum(axis=1).reshape((-1, 1))
            dual_brier = brier_score_loss(y_test, dual_ensemble_predictions[:,1])
            dual_log_loss_val = log_loss(y_test, dual_ensemble_predictions)
            dual_accuracy = accuracy_score(y_test, dual_ensemble_predictions[:,1] > 0.5)
            dual_auc = roc_auc_score(y_test, dual_ensemble_predictions[:,1])
            result_records.append(result_record(algo_name, dual_brier, dual_log_loss_val, dual_accuracy, dual_auc, n_syn_datasets, "Dual"))

    result_df = pd.DataFrame.from_records(result_records)
    result_df.to_csv(str(snakemake.output), index=False)