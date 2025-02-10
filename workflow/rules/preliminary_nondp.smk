from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

n_repeats = 3
max_n_syn_datasets = 10
# n_repeats = 2
# max_n_syn_datasets = 2
config["max_n_syn_datasets"] = max_n_syn_datasets

python_methods = [
    "ddpm", "ddpm-kl", "tvae", 
    "ctgan",
]
synthpop_methods = [
    "synthpop-proper",
    "synthpop-improper"
]
methods = python_methods + synthpop_methods


rule preprocessing_train_test_split:
    output:
        train="results/preliminary-nondp/train-test-splits/repeat_{repeat_ind}/preprocessed-train.csv",
        test="results/preliminary-nondp/train-test-splits/repeat_{repeat_ind}/preprocessed-test.csv",
    threads: 1
    resources:
        mem_mb=1000,
        runtime="10m",
    script:
        "../scripts/preliminary_nondp/train_test_split.py"

rule real_data_results:
    input:
        train="results/preliminary-nondp/train-test-splits/repeat_{repeat_ind}/preprocessed-train.csv",
        test="results/preliminary-nondp/train-test-splits/repeat_{repeat_ind}/preprocessed-test.csv",
    output:
        "results/preliminary-nondp/real-data-results/{repeat_ind}.csv",
    threads: 4
    resources:
        mem_mb=1000,
        runtime="10m",
    script:
        "../scripts/preliminary_nondp/real_data_results.py"

rule aggregate_real_data_results:
    input:
        raw_results=expand(
           "results/preliminary-nondp/real-data-results/{repeat_ind}.csv",
           repeat_ind=range(n_repeats),
        )
    output:
        results="results/preliminary-nondp/real-data-results.csv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime="10m",
    script:
        "../scripts/aggregate_results.py"

rule synthetic_data_python:
    input:
        train="results/preliminary-nondp/train-test-splits/repeat_{repeat_ind}/preprocessed-train.csv",
    output:
        "results/preliminary-nondp/synthetic-data/{method}_{syn_data_ind}_{repeat_ind}.csv",
    threads: 8
    resources:
        mem_mb=4000,
        runtime="10h",
    script:
        "../scripts/preliminary_nondp/synthetic_data.py"

rule synthetic_data_synthpop:
    input:
        train=expand(
            "results/preliminary-nondp/train-test-splits/repeat_{repeat_ind}/preprocessed-train.csv", 
            repeat_ind=range(n_repeats),
        ),
    output:
        proper_syn_data=expand(
            "results/preliminary-nondp/synthetic-data/synthpop-proper_{syn_data_ind}_{repeat_ind}.csv",
            syn_data_ind=range(max_n_syn_datasets), repeat_ind=range(n_repeats),
        ),
        improper_syn_data=expand(
            "results/preliminary-nondp/synthetic-data/synthpop-improper_{syn_data_ind}_{repeat_ind}.csv",
            syn_data_ind=range(max_n_syn_datasets), repeat_ind=range(n_repeats),
        ),
    threads: 4
    resources:
        mem_mb=4000,
        runtime="1h",
    script:
        "../scripts/preliminary_nondp/synthpop.R"

rule learning:
    input:
        test="results/preliminary-nondp/train-test-splits/repeat_{repeat_ind}/preprocessed-test.csv",
        syn_data=[
            "results/preliminary-nondp/synthetic-data/{{method}}_{}_{{repeat_ind}}.csv".format(syn_data_ind)
            for syn_data_ind in range(max_n_syn_datasets)
        ]
    threads: 8
    resources:
        mem_mb=16000,
        runtime="1h",
    output:
        "results/preliminary-nondp/learning-results/{method}_{repeat_ind}.csv"
    script:
        "../scripts/preliminary_nondp/learning.py"

rule aggregate_results:
    input:
        raw_results=expand(
           "results/preliminary-nondp/learning-results/{method}_{repeat_ind}.csv",
           method=methods, repeat_ind=range(n_repeats),
        )
    output:
        results="results/preliminary-nondp/results.csv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime="10m",
    script:
        "../scripts/aggregate_results.py"

rule plots_ready:
    input:
        results="results/preliminary-nondp/results.csv",
        real_data_results="results/preliminary-nondp/real-data-results.csv",
    output:
        "results/preliminary-nondp/ready-to-plot"
    threads: 1
    resources:
        mem_mb=10,
        runtime="1m",
    shell:
        "touch {output}"
