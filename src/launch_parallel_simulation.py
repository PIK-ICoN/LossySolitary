import os

import baobap as bao
import numpy as np

from src.plotting import NewPlot
from src.sim_functions import def_rc_gen, def_rc_gen_global, def_rc_gen_square, def_swing_ob, detect_clusters, \
    gen_swing_rhs, plot_fps


def analyze_multistability(swingpar,
                           sim_dir="simulation_data",
                           f_vars=(10., 1., 20.),
                           run=["asbs", "global", "pp"],
                           node=42,
                           sides=200,
                           nob=1000,
                           times=np.linspace(0, 400, 2000),
                           pp_sparsity=10,
                           debug=True):

    swingpar.time = times

    result_file, batch_dir = bao.prep_dir(sim_dir)

    swing_ob = def_swing_ob(swingpar, times)

    if swingpar.topology == "ensemble":
        rhs = gen_swing_rhs(swingpar, custom_dispatch=True)
    else:
        rhs = gen_swing_rhs(swingpar, custom_dispatch=False)

    alphas = swingpar.alphas
    fps = swingpar.fix_point

    if swingpar.topology == "ensemble":
        fps_dir = os.path.join(sim_dir, "fps/")
        bao.ensure_dir_exists(fps_dir)
    else:
        fps_dir = None

    plot_fps(sim_dir, swingpar)

    co_ob = bao.combine_obs(bao.run_condition_observer, swing_ob)

    bao.save_experiment_script(result_file, __file__)
    bao.save_state_for_analysis(result_file, (sim_dir, swingpar.alphas, swingpar.fix_point, swingpar, node, sides, pp_sparsity))

    brp = bao.BatchRunParameters(number_of_batches=nob,
                                 simulations_per_batch=len(swingpar.alphas),
                                 system_dimension=swingpar.system_dimension)

    # Run Average Single Node Basin Stability analysis:

    if "asbs" in run:
        bao.ensure_dir_exists(os.path.join(sim_dir, "ts_ASBS"))
        asbs_wittness = def_witness(os.path.join(sim_dir, "ts_ASBS"), times, swingpar, debug)

        rc_gen = def_rc_gen(swingpar, fps, alphas, freq_var=f_vars[0], phase_var=np.pi, fps_folder=fps_dir)
        result_file = os.path.join(sim_dir, "results_ASBS.hdf")
        bao.run_experiment(result_file, brp, None, rc_gen, co_ob, times, rhs=rhs, wittness=asbs_wittness, verbose=False)
        bao.pprint(result_file)

    # Run Global Basin Stability analysis:

    if "global" in run and swingpar.topology is not "ensemble":
        bao.ensure_dir_exists(os.path.join(sim_dir, "ts_global"))
        global_wittness = def_witness(os.path.join(sim_dir, "ts_global"), times, swingpar, debug)

        rc_gen = def_rc_gen_global(swingpar, fps, alphas, freq_var=f_vars[1], phase_var=np.pi * f_vars[1])
        result_file = os.path.join(sim_dir, "results_globalBS.hdf")
        bao.run_experiment(result_file, brp, None, rc_gen, co_ob, times, rhs=rhs, wittness=global_wittness, verbose=False)
        bao.pprint(result_file)

    # Run Phase Space Plot

    if "pp" in run and swingpar.topology is not "ensemble":
        rc_gen = def_rc_gen_square(swingpar, sides, sides, fps[::pp_sparsity], alphas[::pp_sparsity],
                                   max_freq=f_vars[2], max_phase=np.pi,
                                   node=node)
        brp.number_of_batches = sides * sides
        brp.simulations_per_batch = len(alphas[::pp_sparsity])
        brp.node = node
        result_file = os.path.join(sim_dir, "results_pp.hdf")
        bao.run_experiment(result_file, brp, None, rc_gen, co_ob, times, rhs=rhs, verbose=False)
        bao.pprint(result_file)

    return sim_dir


def def_witness(ts_dir, times, swingpar, debug):
    def ts_witness(brp, batch, run, batch_folder, states, rc):
        if debug or np.random.randint(10) < 1:
            n = int(states.shape[1] / 2.)
            offset = int(states.shape[0] / 10.)
            natural_frequency = swingpar.input_power / swingpar.damping_coupling
            final_freq = np.mean(states[-offset:, n:], axis=0)
            n_clusters, _, _, _ = detect_clusters(final_freq)
            if n_clusters > 1:
                freq_vari = np.square(states[:, n:] - final_freq)
                max_freq_vari = np.max(freq_vari[-offset:])
                normed_freq = states[:, n:] / natural_frequency

                canvas = NewPlot(output="paper")
                canvas.figure_format = "png"

                filename = os.path.join(ts_dir, "var_b" + str(batch) + "r" + str(run))
                varfig = canvas.space_time_plot(x=np.arange(n), y=times, z=freq_vari,
                                             labels=["node ID", "time", r"$\left(\dot{\phi}_k - \langle\dot{\phi}_k\rangle\right)^2$"],
                                             colorbar=True, sym=False, horizontal=False, zmin=0, zmax=max_freq_vari
                                            )
                canvas.save(filename, varfig)

                filename = os.path.join(ts_dir, "b" + str(batch) + "r" + str(run))
                stfig = canvas.space_time_plot(x=np.arange(n), y=times, z=normed_freq,
                                             labels=["node ID", "time", r"$\frac{D_k\dot{\phi}_k}{\Omega_k}$"],
                                             colorbar=True, sym=True, horizontal=False,
                                             zmin=-1.1, zmax=1.1
                                            )
                canvas.save(filename, stfig)
            return None
    return ts_witness

