# PKS1510_modelling
Code using agnpy for the modelling the FSRQ PKS 1510-1019

## Basic usage

Basic functionalities are wrapped via [click](https://click.palletsprojects.com/en/8.0.x/).
Two commands are available 

### fitting
You can fit a particulare state via the command
```shell 
$ python make.py fit
```
Pass the options `--help` when in doubt
```shell
$ python make.py fit --help
Usage: make.py fit [OPTIONS]

  perform the fit of PKS 1510-089 SED for a given state

Options:
  --state [low|2012|2015a|2015b]
  --k_e FLOAT                     electron normalisation
  --gamma_min FLOAT               minimum Lorentz factor
  --gamma_max FLOAT               maximum Lorentz factor
  --t_var FLOAT                   variability time scale
  --r FLOAT                       distance of blob from BH
  --help                          Show this message and exit.
```

So, to perform the fit of the 2012 adjusting some of the initial parameters (the ones specified in the help command)
```shell
python make.py fit --state 2012 --k_e 0.01 --gamma_min 3 --gamma_max 7e4
```
in the directory `results` a directory per each state will be created, containing the plot of the fitted SED and a dicitonary, a `.yaml` file, containing all the model parameters.

### Plotting
All the states can be plotted together, after they have all been fitted, via the command
```shell
$ python make.py plot 
```
you should obtain a plot liek this

![](results/sed_all_states.png)