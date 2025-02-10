import sys
import os
sys.path.append(os.getcwd())
import pickle
import itertools

import numpy as np 
import pandas as pd 

from lib import rng_initialisation
from lib import experiment_config as ec
from lib import util
from lib.aim.aim import AIM, hypothetical_model_size

from twinify.dataframe_data import DataFrameData

from mbi import Dataset, Domain, FactoredInference, GraphicalModel

if __name__ == "__main__":

    repeat_ind = int(snakemake.wildcards.repeat_ind)
    n_syn_datasets = int(snakemake.wildcards.n_syn_datasets)
    method = str(snakemake.wildcards.method)
    synthetic_data_ind = int(snakemake.wildcards.syn_data_ind)
    dataset = str(snakemake.wildcards.dataset)
    ec.check_dataset_name(dataset)
    seed = rng_initialisation.get_seed(repeat_ind, n_syn_datasets, synthetic_data_ind, method, dataset, "split_synthetic_data")

    np.random.seed(seed)

    train_df = pd.read_csv(str(snakemake.input.train), index_col=False)
    train_df = ec.adult_reduced_discretisation(train_df)

    df_data = DataFrameData(train_df.astype("category"))
    domain_key_list = list(df_data.values_by_col.keys())
    domain_value_count_list = [len(df_data.values_by_col[key]) for key in domain_key_list]
    domain = Domain(domain_key_list, domain_value_count_list)
    data = Dataset(df_data.int_df, domain)

    # aim = AIM(ec.total_epsilon, ec.total_delta, max_model_size=0.001, rho_divider=n_syn_datasets)
    aim = AIM(ec.total_epsilon, ec.total_delta, rho_divider=n_syn_datasets)


    degree = 2
    workload = list(itertools.combinations(data.domain, degree))
    workload = [(cl, 1.0) for cl in workload]

    measurements, synth = aim.run(data, workload)

    syn_df = df_data.int_df_to_cat_df(synth.df)
    syn_df = ec.adult_reduced_discrete_to_continuous(syn_df)
    syn_df.income = (syn_df.income.astype(str) == "True")
    syn_df.to_csv(str(snakemake.output), index=False)