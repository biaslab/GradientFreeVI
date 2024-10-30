# Running the SIR experiments

Navigate to the `sir_model` folder in the command line, and make sure Julia is installed and available to the command line. Then execute
```
julia --project=. nuts_sir.jl
```
to run the experiments and output a .json file containing the results.

# Running the learning rate tuning experiments

Navigate to the `learning_rate` folder in the command line, and make sure Julia and Python are installed and available to the command line. Then execute
```
bash experiments
```
to generate all plots and obtain tuned learning rates and test accuracies.
