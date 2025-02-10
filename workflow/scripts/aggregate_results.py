import sys
import os
sys.path.append(os.getcwd())
import pickle
import itertools

import numpy as np 
import pandas as pd 

if __name__ == "__main__":
    result_dfs = []
    for filename in snakemake.input.raw_results:
        result_dfs.append(pd.read_csv(str(filename), index_col=False))

    df = pd.concat(result_dfs)
    df.to_csv(str(snakemake.output), index=False)
