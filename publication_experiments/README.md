# Setting up the environment
The file `env_freeze.txt` contains the `conda list` output from the environment used in the experiments. It includes more packages than strictly necessary but can serve as a reference for package versions.

# Running the experiments
Run `python main.py` from inside each experiment folder (so the current working directory is set correctly). This will launch the experiment across the two datasets using a single process. Adjust `MAX_PARALLEL_PROCESSES` in `main.py` to enable parallel execution.

The Papermill executor in `main.py` allows tuning two parameters:
- `'max_workers': 6`  
  Maximum number of parallel seeds computed for each dataset.  
- `'n_seeds': 25`  
  Number of seeds for the experiments. The code will try to load `seeds.json` from the experiment folder, and if the number of loaded seeds does not match this parameter, an error is raised.

# Reproducing the experiments from the publication
The file `seeds.json` contains the seeds used in the publication. The experiment automatically loads it if available. Ensure that `n_seeds: 25` is set in main.py, otherwise an error will occur.

The experiments notebook includes a safeguard to prevent overwriting results: it checks for the folder `results_namedataset` and raises an exception if it already exists.

# Additional notebooks
- `plot_onedataset.ipynb`  
  Averages the results of the experiments in `results_namedataset/plots/averaged_results.json`, which also contains the average confusion matrices. It then plots the averaged metrics across seeds in `results_namedataset/plots/namedataset_metric_plot.pdf`.
  
