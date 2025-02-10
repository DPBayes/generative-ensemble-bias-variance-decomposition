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
max_n_syn_datasets = 32
# n_repeats = 2
# max_n_syn_datasets = 2
config["max_n_syn_datasets"] = max_n_syn_datasets

datasets = [
    "abalone", 
    "insurance", 
    "california-housing", 
    "ACS2018"
]

python_methods = [
    "ddpm",
]
synthpop_methods = [
    "synthpop-proper",
]
methods = python_methods + synthpop_methods

variance_estimation_models = [
    "Linear Regression",
    "Ridge Regression",
    "1-NN",
    "5-NN",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "MLP",
    "SVM"
]

rule preprocessing_train_test_split:
    output:
        train="results/regression-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-train.csv",
        test="results/regression-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-test.csv",
        orig_cols="results/regression-datasets/orig_cols/{dataset}/repeat_{repeat_ind}/orig_cols.p",
    threads: 1
    resources:
        mem_mb=1000,
        runtime="10m",
    script:
        "../scripts/train_test_split.py"

rule real_data_results:
    input:
        train="results/regression-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-train.csv",
        test="results/regression-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-test.csv",
        orig_cols="results/regression-datasets/orig_cols/{dataset}/repeat_{repeat_ind}/orig_cols.p",
    output:
        "results/regression-datasets/real-data-results/{dataset}/{repeat_ind}.csv",
    threads: 4
    resources:
        mem_mb=8000,
        runtime="1h",
    script:
        "../scripts/regression-datasets/real_data_results.py"

rule aggregate_real_data_results:
    input:
        raw_results=[
           "results/regression-datasets/real-data-results/{{dataset}}/{}.csv".format(repeat_ind)
           for repeat_ind in range(n_repeats)
        ]
    output:
        results="results/regression-datasets/{dataset}/real-data-results.csv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime="10m",
    script:
        "../scripts/aggregate_results.py"

rule synthetic_data_python:
    input:
        train="results/regression-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-train.csv",
    output:
        "results/regression-datasets/synthetic-data/{dataset}/{method}_{syn_data_ind}_{repeat_ind}.csv",
    threads: 8
    resources:
        mem_mb=12000,
        runtime="1h",
    script:
        "../scripts/synthetic_data.py"

rule synthetic_data_synthpop:
    input:
        train=expand(
            "results/regression-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-train.csv", 
            repeat_ind=range(n_repeats), dataset=datasets
        ),
    output:
        proper_syn_data=expand(
            "results/regression-datasets/synthetic-data/{dataset}/synthpop-proper_{syn_data_ind}_{repeat_ind}.csv",
            syn_data_ind=range(max_n_syn_datasets), repeat_ind=range(n_repeats), dataset=datasets
        ),
    threads: 8
    resources:
        mem_mb=8000,
        runtime="4h",
    shell:
        "Rscript workflow/scripts/regression-datasets/synthpop.R"

rule learning:
    input:
        test="results/regression-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-test.csv",
        orig_cols="results/regression-datasets/orig_cols/{dataset}/repeat_{repeat_ind}/orig_cols.p",
        syn_data="results/regression-datasets/synthetic-data/{dataset}/{method}_{syn_data_ind}_{repeat_ind}.csv"
    output:
        "results/regression-datasets/predictions/{dataset}/{method}_{syn_data_ind}_{repeat_ind}.p"
    threads: 8
    resources:
        mem_mb=16000,
        runtime="1h",
    script:
        "../scripts/regression-datasets/learning.py"

rule ensemble_learning:
    input:
        test="results/regression-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-test.csv",
        orig_cols="results/regression-datasets/orig_cols/{dataset}/repeat_{repeat_ind}/orig_cols.p",
        syn_preds=[
            "results/regression-datasets/predictions/{{dataset}}/{{method}}_{}_{{repeat_ind}}.p".format(syn_data_ind)
            for syn_data_ind in range(max_n_syn_datasets)
        ]
    threads: 4
    resources:
        mem_mb=8000,
        runtime="30m",
    output:
        "results/regression-datasets/learning-results/{dataset}/{method}_{repeat_ind}.csv"
    script:
        "../scripts/regression-datasets/ensemble_learning.py"

rule aggregate_results:
    input:
        raw_results=[
           "results/regression-datasets/learning-results/{{dataset}}/{}_{}.csv".format(method, repeat_ind)
           for method in methods for repeat_ind in range(n_repeats)
        ]
    output:
        results="results/regression-datasets/{dataset}/results.csv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime="10m",
    script:
        "../scripts/aggregate_results.py"

rule variance_estimation:
    input:
        orig_cols="results/regression-datasets/orig_cols/{dataset}/repeat_{repeat_ind}/orig_cols.p",
        test="results/regression-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-test.csv",
        syn_data=[
            "results/regression-datasets/synthetic-data/{{dataset}}/{{method}}_{}_{{repeat_ind}}.csv".format(syn_data_ind)
            for syn_data_ind in range(max_n_syn_datasets)
        ]
    output:
        "results/regression-datasets/variance-estimation/{dataset}/{method}_{model}_{repeat_ind}.p"
    threads: 12
    resources:
        mem_mb=32000,
        runtime="8h"
    script:
        "../scripts/variance_estimation.py"

rule one_large_learning:
    input:
        test="results/regression-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-test.csv",
        orig_cols="results/regression-datasets/orig_cols/{dataset}/repeat_{repeat_ind}/orig_cols.p",
        syn_data="results/regression-datasets/synthetic-data/{dataset}/{method}_0_{repeat_ind}.csv"
    output:
        "results/regression-datasets/one-large-prediction/{dataset}/{method}_{size_mul}_{repeat_ind}.csv"
    threads: 8
    resources:
        mem_mb=16000,
        runtime="1h",
    script:
        "../scripts/regression-datasets/one_large_learning.py"
        
rule aggregate_results_one_large:
    input:
        raw_results=[
           "results/regression-datasets/one-large-prediction/{{dataset}}/{}_{}_{}.csv".format(method, size_mul, repeat_ind)
           for method in methods for size_mul in [1, 2, 4, 5] for repeat_ind in range(n_repeats)
        ]
    output:
        results="results/regression-datasets/{dataset}/one_large_results.csv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime="10m",
    script:
        "../scripts/aggregate_results.py"

rule plots_ready:
    input:
        results=expand("results/regression-datasets/{dataset}/results.csv", dataset=datasets),
        real_data_results=expand("results/regression-datasets/{dataset}/real-data-results.csv", dataset=datasets),
        variance_estimation=expand("results/regression-datasets/variance-estimation/{dataset}/{method}_{model}_{repeat_ind}.p", dataset=datasets, method=methods, model=variance_estimation_models, repeat_ind=range(n_repeats)),
        one_large_results=expand("results/regression-datasets/{dataset}/one_large_results.csv", dataset=datasets),
    output:
        "results/regression-datasets/ready-to-plot"
    threads: 1
    resources:
        mem_mb=10,
        runtime="1m",
    shell:
        "touch {output}"
