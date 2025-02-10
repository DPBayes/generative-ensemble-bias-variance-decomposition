from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

n_repeats = 3
max_n_syn_datasets = 32
# n_repeats = 2
# max_n_syn_datasets = 2
config["max_n_syn_datasets"] = max_n_syn_datasets

datasets = [
    "german-credit", 
    "adult", 
    "breast-cancer"
]

python_methods = [
    "ddpm",
]
synthpop_methods = [
    "synthpop-proper",
]
methods = python_methods + synthpop_methods

variance_estimation_models = [
    "Logistic Regression",
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
        train="results/classification-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-train.csv",
        test="results/classification-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-test.csv",
        orig_cols="results/classification-datasets/orig_cols/{dataset}/repeat_{repeat_ind}/orig_cols.p",
    threads: 1
    resources:
        mem_mb=1000,
        runtime="10m",
    script:
        "../scripts/train_test_split.py"

rule real_data_results:
    input:
        train="results/classification-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-train.csv",
        test="results/classification-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-test.csv",
        orig_cols="results/classification-datasets/orig_cols/{dataset}/repeat_{repeat_ind}/orig_cols.p",
    output:
        "results/classification-datasets/real-data-results/{dataset}/{repeat_ind}.csv",
    threads: 4
    resources:
        mem_mb=8000,
        runtime="1h",
    script:
        "../scripts/classification-datasets/real_data_results.py"

rule aggregate_real_data_results:
    input:
        raw_results=[
           "results/classification-datasets/real-data-results/{{dataset}}/{}.csv".format(repeat_ind)
           for repeat_ind in range(n_repeats)
        ]
    output:
        results="results/classification-datasets/{dataset}/real-data-results.csv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime="10m",
    script:
        "../scripts/aggregate_results.py"

rule synthetic_data_python:
    input:
        train="results/classification-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-train.csv",
    output:
        "results/classification-datasets/synthetic-data/{dataset}/{method}_{syn_data_ind}_{repeat_ind}.csv",
    threads: 8
    resources:
        mem_mb=12000,
        runtime="1h",
    script:
        "../scripts/synthetic_data.py"

rule synthetic_data_synthpop:
    input:
        train=expand(
            "results/classification-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-train.csv", 
            repeat_ind=range(n_repeats), dataset=datasets
        ),
    output:
        proper_syn_data=expand(
            "results/classification-datasets/synthetic-data/{dataset}/synthpop-proper_{syn_data_ind}_{repeat_ind}.csv",
            syn_data_ind=range(max_n_syn_datasets), repeat_ind=range(n_repeats), dataset=datasets
        ),
    threads: 8
    resources:
        mem_mb=8000,
        runtime="4h",
    shell:
        "Rscript workflow/scripts/classification-datasets/synthpop.R"

rule learning:
    input:
        test="results/classification-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-test.csv",
        orig_cols="results/classification-datasets/orig_cols/{dataset}/repeat_{repeat_ind}/orig_cols.p",
        syn_data="results/classification-datasets/synthetic-data/{dataset}/{method}_{syn_data_ind}_{repeat_ind}.csv"
    output:
        "results/classification-datasets/predictions/{dataset}/{method}_{syn_data_ind}_{repeat_ind}.p"
    threads: 8
    resources:
        mem_mb=16000,
        runtime="1h",
    script:
        "../scripts/classification-datasets/learning.py"

rule ensemble_learning:
    input:
        test="results/classification-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-test.csv",
        orig_cols="results/classification-datasets/orig_cols/{dataset}/repeat_{repeat_ind}/orig_cols.p",
        syn_preds=[
            "results/classification-datasets/predictions/{{dataset}}/{{method}}_{}_{{repeat_ind}}.p".format(syn_data_ind)
            for syn_data_ind in range(max_n_syn_datasets)
        ]
    threads: 4
    resources:
        mem_mb=8000,
        runtime="10m",
    output:
        "results/classification-datasets/learning-results/{dataset}/{method}_{repeat_ind}.csv"
    script:
        "../scripts/classification-datasets/ensemble_learning.py"

rule aggregate_results:
    input:
        raw_results=[
           "results/classification-datasets/learning-results/{{dataset}}/{}_{}.csv".format(method, repeat_ind)
           for method in methods for repeat_ind in range(n_repeats)
        ]
    output:
        results="results/classification-datasets/{dataset}/results.csv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime="10m",
    script:
        "../scripts/aggregate_results.py"

rule variance_estimation:
    input:
        orig_cols="results/classification-datasets/orig_cols/{dataset}/repeat_{repeat_ind}/orig_cols.p",
        test="results/classification-datasets/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-test.csv",
        syn_data=[
            "results/classification-datasets/synthetic-data/{{dataset}}/{{method}}_{}_{{repeat_ind}}.csv".format(syn_data_ind)
            for syn_data_ind in range(max_n_syn_datasets)
        ]
    output:
        "results/classification-datasets/variance-estimation/{dataset}/{method}_{model}_{repeat_ind}.p"
    threads: 16
    resources:
        mem_mb=32000,
        runtime="48h"
    script:
        "../scripts/variance_estimation.py"

rule plots_ready:
    input:
        results=expand("results/classification-datasets/{dataset}/results.csv", dataset=datasets),
        real_data_results=expand("results/classification-datasets/{dataset}/real-data-results.csv", dataset=datasets),
        variance_estimation=expand("results/classification-datasets/variance-estimation/{dataset}/{method}_{model}_{repeat_ind}.p", dataset=datasets, method=methods, model=variance_estimation_models, repeat_ind=range(n_repeats))
    output:
        "results/classification-datasets/ready-to-plot"
    threads: 1
    resources:
        mem_mb=10,
        runtime="1m",
    shell:
        "touch {output}"
