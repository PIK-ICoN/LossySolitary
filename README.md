# Lossy Solitaries

Code repository accompanying the publication [[1]](#references): 
_"Network-induced multistability: Lossy coupling and exotic solitary states"_

This project contains the python scripts needed reproduce the numerical experiments 
and respective figures from the paper.

## Requirements

The implementation uses [pyBAOBAP](https://gitlab.pik-potsdam.de/hellmann/pyBAOBAP) 
for MPI-parallelisation. Contact <hellmann@pik-potsdam.de> for support.

Developed and tested on Python >=3.6. Package requirements are listed in the 
`requirements.txt` file. Execute `pip install -r requirements.txt` in a terminal
to install them (though for pyBAOBAP see above link).

## How-To


```
python sampling.py $topology $mode
```

where `topology` is one of

- `northern` for the Scandinacion power grid data set
- `circle` for the symmetric circle topology
- `synthetic` for a single realisation from the  synthetic power grid model [[2]](#references).

and `mode` is 

- `test` for a test run of an experiment to identify bugs
- `debug` for a debug run with a small number of simulations
- ` ` empty for regular simulations

The following simulations are performed for each parameter setup: 
- average single node basin stability, e.g. Fig. 2b in [[1]](#references)
- phase space picture, e.g. Fig. 2a in [[1]](#references)

## Structure

The repository consists of a top-level file `run_numerical_experiment.py`, that 
 has to be executed to run the numerical experiments, and four subfolders:

- The folder `src/` contains source code libraries for the simulation scripts.
- The folder `input_data/` contains the network data for the Scandinavian grid.
- The folder `simulation_data/` is the target of all data output from a simulation. 
- The folder `figures/` contains all generated figure files.


### src

This folder contains library files that are called from `run_numerical_experiment.py`.

- `launch_parallel_simulation.py`: launch a parallel simulation using pyBAOBAP
- `parameters_and_fixed_points.py`: routines to setup network topologies with different parametrisations, fixed point search
- `plotting.py`/`awesomeplot.py`: convenience functions for  plotting
- `rpgm_neonet.py`: generation of [synthetic power grid topologies](https://gitlab.com/luap-public/random-powergrid)
- `sim_functions.py`: misc helper functions to organize data output, define numerical observables or perform data clustering   

### input_data
 
origin and specifics of data, dynamical regimes


- `northern.json`: network topology of the Northern power grid [[1]](#references)
- `area-1.txt`/`area-2.txt`: numerical continuation data, contact Roman Levchenko <rmn01@mail.ru> for detail
- `dynamical_regimes.py`: specifications for the parametrizations used in [[1]](#references)
- `northern_dispatch.py`: allocation of producers and consumers for the Northern power grid, results are stored in `northern_dispatch.csv`

### simulation_data

This folder is empty. Every time a simulation is performed with `run_numerical_experiment.py`,
the output data is stored here. 

### figures

The figures can be recreated using the [Jupyter notebooks](https://jupyter.org/) 
with the respective name. Note that first you need to perform the necessary simulations


## Extended Experiments

### Run selected parameter regimes

You can run a simulation for each dynamical separately, 
here it is`dynamical_regimes[1]` for the Northern power grid,
in the following way:

```
topology = "northern"
dr = dynamical_regimes[1]
single_regime(topology, dr, out_folder="northern_global", number_of_nodes=236, run=["global"])
```

In this example, a global basin stability evaluation is performed.
Pass `run=["asbs"]` for or `run=["pp"]` (or any combination) to obtain
 average single-node basin stability and a phase space portrait, respectively. 

## References

[1]: _"Network-induced multistability through lossy coupling and exotic solitary states"_, DOI:10.1038/s41467-020-14417-7, arXiv:1811.11518 [nlin.AO]

[2]: _"A Random Growth Model for Power Grids and Other Spatially Embedded Infrastructure Networks"_, arXiv:1602.02562 [physics.soc-ph]


