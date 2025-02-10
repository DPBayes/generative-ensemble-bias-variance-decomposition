
n_repeats = 3
n_syn_datasets_values = [1, 2, 4, 8, 16, 32]
# n_repeats = 2
# n_syn_datasets_values = [1, 2]

max_n_syn_datasets = max(n_syn_datasets_values)
config["max_n_syn_datasets"] = max_n_syn_datasets
config["n_syn_datasets_values"] = n_syn_datasets_values

datasets = ["adult-reduced"]
split_methods = ["AIM"]
non_split_methods = ["NAPSU-MQ"]
methods = split_methods + non_split_methods

rule preprocessing_train_test_split:
    output:
        train="results/dp-experiment/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-train.csv",
        test="results/dp-experiment/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-test.csv",
        orig_cols="results/dp-experiment/orig_cols/{dataset}/repeat_{repeat_ind}/orig_cols.p",
    threads: 1
    resources:
        mem_mb=1000,
        runtime="10m",
    script:
        "../scripts/train_test_split.py"

rule real_data_results:
    input:
        train="results/dp-experiment/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-train.csv",
        test="results/dp-experiment/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-test.csv",
        orig_cols="results/dp-experiment/orig_cols/{dataset}/repeat_{repeat_ind}/orig_cols.p",
    output:
        "results/dp-experiment/real-data-results/{dataset}/{repeat_ind}.csv",
    threads: 4
    resources:
        mem_mb=8000,
        runtime="1h",
    script:
        "../scripts/classification-datasets/real_data_results.py"

rule aggregate_real_data_results:
    input:
        raw_results=[
           "results/dp-experiment/real-data-results/{{dataset}}/{}.csv".format(repeat_ind)
           for repeat_ind in range(n_repeats)
        ]
    output:
        results="results/dp-experiment/{dataset}/real-data-results.csv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime="10m",
    script:
        "../scripts/aggregate_results.py"

rule split_syn_data:
    input:
        train="results/dp-experiment/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-train.csv",
    output:
        "results/dp-experiment/synthetic-data/{dataset}/{method}_{n_syn_datasets}_{syn_data_ind}_{repeat_ind}.csv"
    wildcard_constraints:
        method="AIM"
    threads: 8
    resources:
        mem_mb=8000,
        runtime="4h",
    script:
        "../scripts/dp-experiment/split_synthetic_data.py"
    
rule non_split_generator:
    input:
        train="results/dp-experiment/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-train.csv",
    output:
        "results/dp-experiment/generators/{dataset}/{method}_{repeat_ind}.p"
    threads: 16
    resources:
        mem_mb=8000,
        runtime="48h",
    script:
        "../scripts/dp-experiment/non_split_generator.py"


rule non_split_syn_data:
    input:
        generator="results/dp-experiment/generators/{dataset}/{method}_{repeat_ind}.p"
    output:
        ["results/dp-experiment/synthetic-data/{{dataset}}/{{method}}_{}_{}_{{repeat_ind}}.csv".format(n_syn_datasets, syn_data_ind)
        for n_syn_datasets in n_syn_datasets_values for syn_data_ind in range(n_syn_datasets)]
    threads: 16
    resources:
        mem_mb=8000,
        runtime="4h",
    script:
        "../scripts/dp-experiment/non_split_synthetic_data.py"

ruleorder: split_syn_data > non_split_syn_data

rule learning:
    input:
        test="results/dp-experiment/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-test.csv",
        orig_cols="results/dp-experiment/orig_cols/{dataset}/repeat_{repeat_ind}/orig_cols.p",
        syn_data="results/dp-experiment/synthetic-data/{dataset}/{method}_{n_syn_datasets}_{syn_data_ind}_{repeat_ind}.csv"
    output:
        "results/dp-experiment/predictions/{dataset}/{method}_{n_syn_datasets}_{syn_data_ind}_{repeat_ind}.p"
    threads: 8
    resources:
        mem_mb=8000,
        runtime="1h",
    script:
        "../scripts/dp-experiment/learning.py"

rule ensemble_learning:
    input:
        test="results/dp-experiment/train-test-splits/{dataset}/repeat_{repeat_ind}/preprocessed-test.csv",
        orig_cols="results/dp-experiment/orig_cols/{dataset}/repeat_{repeat_ind}/orig_cols.p",
        syn_preds=lambda wildcards: [
            "results/dp-experiment/predictions/{{dataset}}/{{method}}_{{n_syn_datasets}}_{}_{{repeat_ind}}.p".format(syn_data_ind)
            for syn_data_ind in range(int(wildcards.n_syn_datasets))
        ]
    threads: 4
    resources:
        mem_mb=8000,
        runtime="10m",
    output:
        "results/dp-experiment/learning-results/{dataset}/{method}_{n_syn_datasets}_{repeat_ind}.csv"
    script:
        "../scripts/dp-experiment/ensemble_learning.py"

rule aggregate_results:
    input:
        raw_results=[
           "results/dp-experiment/learning-results/{{dataset}}/{}_{}_{}.csv".format(method, n_syn_datasets, repeat_ind)
           for method in methods for n_syn_datasets in n_syn_datasets_values for repeat_ind in range(n_repeats)
        ]
    output:
        results="results/dp-experiment/{dataset}/results.csv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime="10m",
    script:
        "../scripts/aggregate_results.py"

rule plots_ready:
    input:
        results=expand("results/dp-experiment/{dataset}/results.csv", dataset=datasets),
        real_data_results=expand("results/dp-experiment/{dataset}/real-data-results.csv", dataset=datasets),
    output:
        "results/dp-experiment/ready-to-plot"
    threads: 1
    resources:
        mem_mb=10,
        runtime="1m",
    shell:
        "touch {output}"

