import sys
import os
sys.path.append(os.getcwd())
import pickle

import numpy as np 
import pandas as pd 
import torch

import synthcity.logger as log
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

from lib import rng_initialisation
from lib import experiment_config as ec


if __name__ == "__main__":

    repeat_ind = int(snakemake.wildcards.repeat_ind)
    method = str(snakemake.wildcards.method)
    synthetic_data_ind = int(snakemake.wildcards.syn_data_ind)
    dataset = str(snakemake.wildcards.dataset)
    ec.check_dataset_name(dataset)
    seed = rng_initialisation.get_seed(repeat_ind, synthetic_data_ind, method, dataset, "synthetic_data")

    np.random.seed(seed)
    torch.manual_seed(seed)

    data = pd.read_csv(str(snakemake.input.train), index_col=False)
    loader = GenericDataLoader(data)

    # !!! SMOKETEST CODE !!!
    # if method == "ddpm":
    #     syn_model = Plugins().get("ddpm", n_iter=100, is_classification=ec.is_classification(dataset), gaussian_loss_type="mse")
    # else:
    #     raise Exception("Unknown method {}".format(method))
    # !!! SMOKETEST CODE !!!

    # Real Code
    if method == "ddpm":
        syn_model = Plugins().get("ddpm", is_classification=ec.is_classification(dataset), gaussian_loss_type="mse")
    else:
        raise Exception("Unknown method {}".format(method))

    syn_model.fit(loader)
    n_syn_dataset = data.shape[0]
    # n_syn_dataset = 100
    syn_data = syn_model.generate(n_syn_dataset * ec.n_generated_syn_datasets)
    syn_data.dataframe().to_csv(str(snakemake.output), index=False)
    
