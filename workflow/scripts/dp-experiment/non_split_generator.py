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
from lib.aim.aim import AIM

from mbi import Dataset, Domain, FactoredInference, GraphicalModel

from twinify.dataframe_data import DataFrameData
from twinify.dataframe_data import DataFrameData
from twinify.napsu_mq.undirected_graph import UndirectedGraph
from twinify.napsu_mq import maximum_entropy_inference as mei
from twinify.napsu_mq.markov_network import MarkovNetwork
from twinify.napsu_mq.marginal_query import FullMarginalQuerySet
from twinify.napsu_mq import privacy_accounting
from twinify.napsu_mq import NapsuMQResult

import jax.numpy as jnp
import jax
import d3p.random

import numpyro.diagnostics


if __name__ == "__main__":

    repeat_ind = int(snakemake.wildcards.repeat_ind)
    method = str(snakemake.wildcards.method)
    dataset = str(snakemake.wildcards.dataset)
    ec.check_dataset_name(dataset)
    seed = rng_initialisation.get_seed(repeat_ind, method, dataset, "non_split_generator")

    napsu_rng = jax.random.PRNGKey(seed)
    np.random.seed(seed)

    train_df = pd.read_csv(str(snakemake.input.train), index_col=False)
    train_df = ec.adult_reduced_discretisation(train_df)

    df_data = DataFrameData(train_df.astype("category"))
    domain_key_list = list(df_data.values_by_col.keys())
    domain_value_count_list = [len(df_data.values_by_col[key]) for key in domain_key_list]
    domain = Domain(domain_key_list, domain_value_count_list)
    data = Dataset(df_data.int_df, domain)

    aim = AIM(ec.napsu_query_selection_epsilon, ec.napsu_query_selection_delta, max_model_size=0.001)

    degree = 2
    workload = list(itertools.combinations(data.domain, degree))
    workload = [(cl, 1.0) for cl in workload]

    measurements = aim.run(data, workload, return_syn_data=False)


    selected_queries = [measurement[3] for measurement in measurements]
    napsu_queries = FullMarginalQuerySet(selected_queries, df_data.values_by_col)
    canon_queries = napsu_queries.get_canonical_queries()
    query_values = jnp.sum(canon_queries.flatten()(df_data.int_df.to_numpy()), axis=0)
    n_train = df_data.int_df.shape[0]

    napsu_epsilon = ec.napsu_measurement_epsilon
    napsu_delta = ec.napsu_measurement_delta
    sensitivity = np.sqrt(2 * len(canon_queries.queries.keys()))
    sigma_DP = privacy_accounting.sigma(napsu_epsilon, napsu_delta, sensitivity)

    napsu_rng, noise_rng = jax.random.split(napsu_rng)
    dp_noise = jax.random.normal(noise_rng, (query_values.shape[0],)) * sigma_DP
    dp_query_values = query_values + dp_noise

    network = MarkovNetwork(df_data.values_by_col, canon_queries)

    approx_rng, approx_sample_rng, mcmc_rng = jax.random.split(napsu_rng, 3)
    mcmc = mei.run_numpyro_mcmc(mcmc_rng, dp_query_values, n_train, sigma_DP, network, num_samples=1000, num_warmup=500, num_chains=2)
    # mcmc = mei.run_numpyro_mcmc(mcmc_rng, dp_query_values, n_train, sigma_DP, network, num_samples=10, num_warmup=5, num_chains=2)

    diagnostics = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True),group_by_chain=True)

    posterior_values = mcmc.get_samples(group_by_chain=False)["lambdas"]
    napsu_result = NapsuMQResult(df_data.values_by_col, canon_queries, posterior_values, df_data.data_description)

    save_obj = {
        "result": napsu_result,
        "diagnostics": diagnostics,
        "n_train": n_train,
    }

    with open(str(snakemake.output), "wb") as file:
        pickle.dump(save_obj, file)