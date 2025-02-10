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


if __name__ == "__main__":

    repeat_ind = int(snakemake.wildcards.repeat_ind)
    method = str(snakemake.wildcards.method)
    synthetic_data_ind = int(snakemake.wildcards.syn_data_ind)
    seed = rng_initialisation.get_seed(repeat_ind, synthetic_data_ind, method, "california-housing", "synthetic_data")
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = pd.read_csv(str(snakemake.input.train), index_col=False)
    loader = GenericDataLoader(data.astype(np.float32), target_column="LogMedHouseVal")

    # !!! SMOKETEST CODE !!!
    # if method == "ddpm":
    #     syn_model = Plugins().get(method, is_classification=False, n_iter=2)
    # if method == "arf":
    #     syn_model = Plugins().get(method, max_iters=2)
    # else:
    #     syn_model = Plugins().get(method, n_iter=2)
    # !!! SMOKETEST CODE !!!

    # Real Code
    if method == "ddpm":
        syn_model = Plugins().get("ddpm", is_classification=False, gaussian_loss_type="mse")
    elif method == "ddpm-kl":
        syn_model = Plugins().get("ddpm", is_classification=False, gaussian_loss_type="kl")
    elif method == "arf":
        syn_model = Plugins().get(method, min_node_size=10)
    else:
        syn_model = Plugins().get(method)

    syn_model.fit(loader)
    n_syn_dataset = data.shape[0]
    syn_data = syn_model.generate(n_syn_dataset)
    syn_data.dataframe().to_csv(str(snakemake.output), index=False)
    
