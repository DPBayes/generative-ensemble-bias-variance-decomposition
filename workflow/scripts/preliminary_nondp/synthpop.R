
library(synthpop)


# n_repeats = 2
# max_n_syn_datasets = 2
n_repeats = 3
max_n_syn_datasets = 10

set.seed(585638)

for (repeat_ind in 1:n_repeats){
    data_filename = sprintf("results/preliminary-nondp/train-test-splits/repeat_%d/preprocessed-train.csv", repeat_ind - 1)
    data = read.obs(data_filename)
    syn_data_proper = syn(data, proper=TRUE, m=max_n_syn_datasets)
    syn_data_improper = syn(data, proper=FALSE, m=max_n_syn_datasets)

    for (syn_data_ind in 1:max_n_syn_datasets){
        output_proper = sprintf("results/preliminary-nondp/synthetic-data/synthpop-proper_%d_%d.csv", syn_data_ind - 1, repeat_ind - 1)
        output_improper = sprintf("results/preliminary-nondp/synthetic-data/synthpop-improper_%d_%d.csv", syn_data_ind - 1, repeat_ind - 1)

        write.csv(syn_data_proper$syn[[syn_data_ind]], output_proper, row.names=FALSE)
        write.csv(syn_data_improper$syn[[syn_data_ind]], output_improper, row.names=FALSE)
    }
}
