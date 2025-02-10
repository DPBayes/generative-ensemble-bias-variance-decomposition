import sys
import os
sys.path.append(os.getcwd())
import pickle

import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
import sklearn.datasets

from lib import rng_initialisation

if __name__ == "__main__":

    repeat_ind = int(snakemake.wildcards.repeat_ind)
    seed = rng_initialisation.get_seed(repeat_ind, "california-housing", "train_test_split")
    np.random.seed(seed)

    bunch = sklearn.datasets.fetch_california_housing(as_frame=True, data_home="./datasets")
    X = bunch.data 
    X["LogPopulation"] = np.log(X.Population)
    X["LogMedInc"] = np.log(X.MedInc)
    X.drop(columns=["Population", "MedInc"], inplace=True)
    y = np.log(bunch.target)
    y.name = "LogMedHouseVal"

    all_data = pd.concat((X, y), axis=1)
    all_data = all_data[all_data.AveOccup < 30]
    all_data = all_data[all_data.AveRooms < 50]
    data, test_data = train_test_split(all_data, test_size=0.25)

    data.to_csv(str(snakemake.output.train), index=False)
    test_data.to_csv(str(snakemake.output.test), index=False)