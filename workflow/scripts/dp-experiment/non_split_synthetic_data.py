import sys
import os
sys.path.append(os.getcwd())
import pickle

import numpy as np 
import pandas as pd 

from lib import rng_initialisation
from lib import experiment_config as ec
from lib import util

from twinify.dataframe_data import DataFrameData
from twinify.napsu_mq import NapsuMQResult
import d3p.random


if __name__ == "__main__":

    repeat_ind = int(snakemake.wildcards.repeat_ind)
    method = str(snakemake.wildcards.method)
    dataset = str(snakemake.wildcards.dataset)
    ec.check_dataset_name(dataset)
    seed = rng_initialisation.get_seed(repeat_ind, method, dataset, "non_split_synthetic_data")

    np.random.seed(seed)
    generation_rng = d3p.random.PRNGKey(seed)

    with open(str(snakemake.input), "rb") as file:
        load_obj = pickle.load(file)

    n_syn_dataset_values = snakemake.config["n_syn_datasets_values"]
    base_filename = "results/dp-experiment/synthetic-data/{}/{}_{}_{}_{}.csv"

    syn_df_dict = {n_syn_datasets: [] for n_syn_datasets in n_syn_dataset_values}
    if method == "NAPSU-MQ":
        napsu_result = load_obj["result"]
        n_train = load_obj["n_train"]
        for n_syn_datasets in n_syn_dataset_values:
            syn_df_dict[n_syn_datasets] = napsu_result.generate(generation_rng, n_syn_datasets, n_train, single_dataframe=False)
    else:
        raise Exception("Unknown method {}".format(method))



    for n_syn_datasets, syn_df_list in syn_df_dict.items():
        for i, syn_df in enumerate(syn_df_list):
            syn_df = ec.adult_reduced_discrete_to_continuous(syn_df)
            syn_df.income = (syn_df.income.astype(str) == "True")
            syn_df.to_csv(base_filename.format(dataset, method, n_syn_datasets, i, repeat_ind), index=False)