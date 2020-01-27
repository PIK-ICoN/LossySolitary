"""
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
"""

import os
import sys

import numpy as np
from baobap import ensure_dir_exists, pprint

from input_data import dynamical_regimes
from src.launch_parallel_simulation import analyze_multistability
from src.parameters_and_fixed_points import setup_circle, setup_ensemble, setup_northern
from src.plotting import clustering_frequency_data
from src.sim_functions import find_sim_dir


# ------------------------------------------------------------------------------------------------------------------- #


def single_regime(topology, dr, out_folder, node=0, number_of_nodes=48, max_alpha=0.3, run=["asbs", "pp"], debug=False,
                  test=False):
    name = ("cp_{}_PK_{:.3f}_D2KH_{:.3f}").format(*dr.values()).rstrip('.0').replace(".", "p")

    if debug:
        alphas = np.linspace(0., max_alpha, 3)
        sides = 3
        nob = 4
        times = np.linspace(0, 100, 1000)
        pp_sparsity = 2
        sim_dir = find_sim_dir(os.path.join("simulation_data", "debug", out_folder), name)
    elif test:
        alphas = np.linspace(0., max_alpha, 31)
        sides = 25
        nob = 100
        times = np.linspace(0, 100, 1000)
        pp_sparsity = 10
        sim_dir = find_sim_dir(os.path.join("simulation_data", "test", out_folder), name)
    else:
        alphas = np.linspace(0., max_alpha, 61)
        sides = 200
        nob = 1000
        times = np.linspace(0, 400, 2000)
        pp_sparsity = 10
        sim_dir = find_sim_dir(os.path.join("simulation_data", out_folder), name)

    ensure_dir_exists(sim_dir)
    pprint("saving results for {} to: {}".format(topology, sim_dir))

    if topology == "synthetic":
        swingpar = setup_ensemble(dr, alphas, sim_dir, number_of_nodes=number_of_nodes, sample_size=100)
        swingpar.topology = "synthetic"
    elif topology == "circle":
        swingpar = setup_circle(dr, alphas, sim_dir, number_of_nodes=number_of_nodes)
    elif topology == "northern":
        swingpar = setup_northern(dr, alphas, sim_dir)
    else:
        raise ValueError("Wrong topology type.")

    sim_dir = analyze_multistability(swingpar,
                                     sim_dir=sim_dir,
                                     f_vars=(10., 1., 20.),
                                     sides=sides,
                                     nob=nob,
                                     times=times,
                                     pp_sparsity=pp_sparsity,
                                     run=run,
                                     node=node,
                                     debug=debug)

    for experiment in run:
        if experiment == "asbs":
            clustering_frequency_data(os.path.join(sim_dir, "results_ASBS/"), os.path.join(sim_dir, "results_ASBS.hdf"),
                                      swingpar)
        elif experiment == "pp":
            clustering_frequency_data(os.path.join(sim_dir, "results_pp/"), os.path.join(sim_dir, "results_pp.hdf"),
                                      swingpar)


def analyze_all_regimes(topology, out_folder, node=0, number_of_nodes=48, run=["asbs", "pp"], test=False, debug=False,
                        idx=None):
    # run all experiments
    if idx is None:
        for dr in dynamical_regimes:
            single_regime(topology, dr, out_folder=out_folder, node=node, number_of_nodes=number_of_nodes, run=run,
                          debug=debug, test=test)
    # run a single experiment
    elif type(idx) == int:
        single_regime(topology, dynamical_regimes[idx], out_folder=out_folder, node=node,
                      number_of_nodes=number_of_nodes, run=run, debug=debug, test=test)
    # run any series of experiments
    else:
        for dr in [dynamical_regimes[k] for k in idx]:
            single_regime(topology, dr, out_folder=out_folder, node=node, number_of_nodes=number_of_nodes, run=run,
                          debug=debug, test=test)


# ------------------------------------------------------------------------------------------------------------------- #
# numerical experiments

def circle_analysis(debug=True, test=False, idx=None):
    topology = "circle"
    analyze_all_regimes(topology, out_folder="resubmission/{}/".format(topology), node=0,
                        number_of_nodes=48, run=["asbs", "pp"], test=test, debug=debug, idx=idx)


def northern_analysis(debug=True, test=False, idx=None):
    topology = "northern"
    analyze_all_regimes(topology, out_folder="resubmission/{}/".format(topology), node=0,
                        number_of_nodes=236, run=["asbs", "pp"], test=test, debug=debug, idx=idx)


def synthetic_analysis(debug=True, test=False, idx=None):
    topology = "synthetic"
    analyze_all_regimes(topology, out_folder="resubmission/{}/".format(topology), node=0,
                        number_of_nodes=236, run=["asbs"], test=test, debug=debug, idx=idx)


def single_debug_run(topology="circle", debug=True, test=False, run=["asbs", "pp"], number_of_nodes=48):
    single_regime(topology,
                  {"consumer/producer": 1, "P/K": 1. / 6, "D^2/KH": 0.1 ** 2},
                  number_of_nodes=number_of_nodes,
                  out_folder="resubmission/{}/".format(topology),
                  run=run,
                  test=test,
                  debug=debug)


# ------------------------------------------------------------------------------------------------------------------- #
# add numerical experiment to be executed

if __name__ == "__main__":

    debug = False
    test = False

    if len(sys.argv) > 2 and sys.argv[2] == "test":
        test = True
    if len(sys.argv) > 2 and sys.argv[2] == "debug":
        debug = True

    if sys.argv[1] == "circle":
        circle_analysis(debug=debug, test=test)
    elif sys.argv[1] == "northern":
        northern_analysis(debug=debug, test=test)
    elif sys.argv[1] == "synthetic":
        synthetic_analysis(debug=debug, test=test)
    else:
        raise ValueError("invalid input.")
