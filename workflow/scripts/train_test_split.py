import sys
import os
sys.path.append(os.getcwd())
import pickle
import re

import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
import sklearn.datasets

from lib import rng_initialisation
from lib import util
from lib import experiment_config as ec

if __name__ == "__main__":

    repeat_ind = int(snakemake.wildcards.repeat_ind)
    dataset = str(snakemake.wildcards.dataset)
    seed = rng_initialisation.get_seed(repeat_ind, dataset, "train_test_split")
    np.random.seed(seed)

    target_name = ec.target_names[dataset]

    if dataset == "california-housing":
        bunch = sklearn.datasets.fetch_california_housing(as_frame=True, data_home="./datasets")
        X = bunch.data 
        X["LogPopulation"] = np.log(X.Population)
        X["LogMedInc"] = np.log(X.MedInc)
        X.drop(columns=["Population", "MedInc"], inplace=True)
        y = np.log(bunch.target)
        y.name = "LogMedHouseVal"

        df = pd.concat((X, y), axis=1)
        df = df[df.AveOccup < 30]
        df = df[df.AveRooms < 50]
    elif dataset == "abalone":
        df = pd.read_csv("datasets/abalone.csv", index_col=False)
    elif dataset == "insurance":
        df = pd.read_csv("datasets/insurance.csv", index_col=False)
        df["log_charges"] = np.log(df.charges)
        df.drop(columns=["charges"], inplace=True)
    elif dataset == "ACS2018":
        df = pd.read_csv("datasets/ACS-2018.csv", index_col=False)
    elif dataset == "german-credit":
        col_names = [
            "AccountStatus", "Duration", "CreditHistory", "Purpose", "CreditAmount", "SavingAccount",
            "PresentEmployment", "InstallmentRate", "PersonalStatus", "OtherDebtors", 
            "PresentResident", "Property", "Age", "OtherPlans", "Housing", "ExistingCredits",
            "Job", "LiablePeople", "Telephone", "Foreign", "Target"
        ]
        df = pd.read_csv("datasets/statlog+german+credit+data/german.data", index_col=False, sep=" ", header=None, names=col_names)
        df.Target = df.Target.apply(lambda t: "Good" if t == 1 else "Bad")
        df["TargetGood"] = (df.Target == "Good")
        df.drop(columns=["Target"], inplace=True)
    elif dataset == "adult":
        df = pd.read_csv("datasets/adult.csv", na_values="?")
        df.income = df.income.apply(lambda inc: inc == "<=50K")
        df.drop(columns=["fnlwgt", "educational-num"], inplace=True)
        df.dropna(axis="index", inplace=True)
    elif dataset == "adult-reduced":
        df = pd.read_csv("datasets/adult.csv", na_values="?")
        df.income = df.income.apply(lambda inc: inc == "<=50K")
        df.drop(columns=["fnlwgt", "educational-num", "occupation", "relationship", "native-country"], inplace=True)
        df.dropna(axis="index", inplace=True)
        df["capital-gain"] = (df["capital-gain"] > 0)
        df["capital-loss"] = (df["capital-loss"] > 0)
    elif dataset == "breast-cancer":
        bunch = sklearn.datasets.load_breast_cancer(as_frame=True)
        df = bunch.frame 
        df.target = df.target.astype(bool)
    else:
        raise Exception("Uknown dataset {}".format(dataset))

    df.columns = [re.sub("\W", "_", s) for s in df.columns]
    oh_df = util.get_oh_df(df, target_name)
    orig_cols = list(oh_df.columns)
    data, test_data = train_test_split(df, test_size=0.25)

    data.to_csv(str(snakemake.output.train), index=False)
    test_data.to_csv(str(snakemake.output.test), index=False)

    with open(str(snakemake.output.orig_cols), "wb") as file:
        pickle.dump(orig_cols, file)