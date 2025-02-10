
library(synthpop)


# n_repeats = 2
# max_n_syn_datasets = 2
n_repeats = 3
max_n_syn_datasets = 32
n_generated_syn_datasets = 5
datasets = c("california-housing", "abalone", "insurance", "ACS2018")

set.seed(7605870)

for (dataset in datasets){
    for (repeat_ind in 1:n_repeats){
        data_filename = sprintf("results/regression-datasets/train-test-splits/%s/repeat_%d/preprocessed-train.csv", dataset, repeat_ind - 1)
        data = read.obs(data_filename)
        syn_data_proper = syn(data, proper=TRUE, m=max_n_syn_datasets, k=nrow(data) * n_generated_syn_datasets)

        for (syn_data_ind in 1:max_n_syn_datasets){
            output_proper = sprintf("results/regression-datasets/synthetic-data/%s/synthpop-proper_%d_%d.csv", dataset, syn_data_ind - 1, repeat_ind - 1)
            write.csv(syn_data_proper$syn[[syn_data_ind]], output_proper, row.names=FALSE)
        }
    }
}