# Installing Dependencies

We use [Poetry](https://python-poetry.org/) package manager for Python to manage 
dependencies. After installing Poetry according to their instructions, run 
```bash
poetry install
```
to install the python dependencies.

## R Dependencies

We use the R library [Synthpop](https://synthpop.org.uk/) as one synthetic 
data generation algorithm. It can be installed by starting R with 
```
R
```
and running 
```R
install.packages("synthpop")
```

# Datasets
The datasets are included in this submission. The code we used to download the 
ACS 2018 dataset is in the notebook `download_ACS_2018.ipynb`.

# Running the Code

We use [Snakemake](https://snakemake.github.io/) to run the experiments.
The command 
```
poetry run snakemake -j 5
```
will run all of the experiments with 5 jobs in parallel. This can take several 
days if run a single computer.

After the code has run, the notebooks in the `plotting` directory can be run to plot the figures, which will 
appear in the `figures` directory and its subdirectories. 

The code for the random forest experiment is in the notebook
`random-forest-mse-prediction.ipynb`.