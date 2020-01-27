import copy
import os
import sys

import baobap as bao
import numpy as np
import scipy.sparse

from src.sim_functions import SwingParameters, gen_swing_rhs


def draw_grid_from_neonet(number_of_nodes):
    from src.rpgm_neonet import RpgAlgorithm as NeoNet

    Y = scipy.sparse.lil_matrix((number_of_nodes, number_of_nodes), dtype=np.complex128)

    g = NeoNet(L=1)
    g.set_params(n=np.array([number_of_nodes], dtype=int),
                 n0=np.array([1], dtype=int), #int(0.8*number_of_nodes)
                 w=[1.],  # 1
                 p=[.2],  # .2
                 q=[.3],  # .3
                 r=[1. / 3.],  # 1/3
                 s=[0.1],  # .1
                 u=[1e-6],  # this should be practically 0.
                 alpha=0.,
                 beta=0.,
                 gamma=1.
                 )
    g.prepare()
    g.initialise(0)
    g.grow(0)
    for e in g.levgraph[0].get_edgelist():
        dist = g._get_distances([e[0]], [e[1]])[0, 0]
        Y[e[0], e[1]] = Y[e[1], e[0]] = 1. / dist
    return Y, g

# this should not be run_on_master as it will be called in each batch for the ensemble
def random_admittance(number_of_nodes, draw_grid=draw_grid_from_neonet):
    # catch errors in network construction
    tries = 10
    for i in range(tries):
        try:
            Y, g = draw_grid(number_of_nodes)
        except:
            if i < tries - 1:  # i is zero indexed
                continue
            else:
                raise
        break

    degree = Y.dot(np.ones(number_of_nodes))

    # set diagonal
    for i in range(number_of_nodes):
        Y[i, i] = - degree[i]

    locations = {i: d for (i, d) in enumerate(zip(g.lat, g.lon))}

    return Y, locations

def northern_admittance(filename):
    import json

    with open(filename) as f:
        data = json.load(f)

    number_of_nodes = len(data["nodes"])

    # coupling matrix
    Y = scipy.sparse.lil_matrix((number_of_nodes, number_of_nodes), dtype=np.complex128)

    # The json file needs to contain "links" with "source" and "target" and a
    # "coupling" field.
    for e in data["links"]:
        Y[e["source"], e["target"]] = e["coupling"]
        Y[e["target"], e["source"]] = e["coupling"]

    degree = Y.dot(np.ones(number_of_nodes))

    # set Laplacian diagonal
    for i in range(number_of_nodes):
        Y[i, i] = - degree[i]

    return Y

def circle_admittance(n, coupling_range):
    Y = scipy.sparse.lil_matrix((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(-coupling_range, coupling_range + 1):
            if not j == 0:
                Y[i, (i + j) % n] = 6.
                Y[(i + j) % n, i] = 6.

    degree = Y.dot(np.ones(n))

    # set Laplacian diagonal
    for i in range(n):
        Y[i, i] = - degree[i]

    return Y

# this should not be run_on_master as it will be called in each batch for the ensemble
def determine_synchronizable_grid(P_init, Y_norm, number_of_nodes, sample_size=100):
    """
    draw $sample_size viable network/dispatch pairs and select the most synchronisable one
    :param P:
    :param Y_norm:
    :param number_of_nodes:
    :param max_trials:
    :param sample_size:
    :return:
    """
    counter = 0
    scores = []
    Ys = []
    Ps = []
    locs = []
    while counter < sample_size:
        P = np.random.permutation(P_init)
        Y, pos = random_admittance(number_of_nodes)
        Y *= 1.j
        Y /= 1. * Y_norm * np.mean(np.abs(Y)[Y.nonzero()])
        new_score = sync_condition(Y, P)
        counter += 1
        if new_score < 1:
            scores.append(new_score)
            Ys.append(copy.deepcopy(Y))
            Ps.append(copy.deepcopy(P))
            locs.append(copy.deepcopy(pos))
    if len(scores) > 0:
        idx = np.argmin(scores)
    else:
        print("No further candidates found. Try to increase max_trials.")
        idx = 0

    return Ys[idx], Ps[idx], scores[idx]

# setups should only run on the master process
@bao.run_on_master
def setup_ensemble(dynamical_regime, alphas, out_folder, number_of_nodes=100, sample_size=100):
    cp = dynamical_regime["consumer/producer"]
    pk = dynamical_regime["P/K"]
    dkh = dynamical_regime["D^2/KH"]

    swingpars = SwingParameters(number_of_nodes)
    swingpars.topology = "ensemble"
    swingpars.alphas = alphas

    # Setting H = 1 so that P, K and D are directly the relevant quantities.
    H = 1.

    # damping
    swingpars.damping_coupling = np.sqrt(dkh) / H

    # net power
    assert number_of_nodes % (cp + 1) == 0
    n_gen = int(number_of_nodes / (cp + 1))
    Pg = (cp + 1.) / 2.
    Pc = - 1. * Pg / cp
    P_init = np.ones(number_of_nodes) * Pc
    idx = np.random.choice(range(number_of_nodes), n_gen, replace=False)
    P_init[idx] = Pg

    Y, P, score = determine_synchronizable_grid(P_init, pk, number_of_nodes, sample_size=sample_size)

    swingpars.phase_coupling = scipy.sparse.csr_matrix(Y.todense())
    swingpars.input_power = P

    average_coupling = np.mean(np.abs(Y)[Y.nonzero()])
    min_coupling = np.min(np.abs(Y)[Y.nonzero()])
    max_coupling = np.max(np.abs(Y)[Y.nonzero()])
    degree = np.array([Y[i, i] for i in range(number_of_nodes)])

    print("-----------------------")
    print("Set up parameters as:")
    print(r"$\alpha$ ranges from {} to {}".format(alphas[0], alphas[-1]))
    print(
        r"${}\phi'' + {:.2f}\phi' + \sum_j ({} + dK_ij) sin(\phi_i - \phi_j -\alpha) = {:.2f} (prod) / {:.2f} (con)$".format(
            H, swingpars.damping_coupling, average_coupling, -cp * Pc, Pc))
    print("distr. of K_ij: min={:.2f} mean={:.2f} max={:.2f}".format(min_coupling, average_coupling, max_coupling))
    print("distr. of degree: min={:.2f} mean={:.2f} max={:.2f}".format(abs(degree.min()),
                                                                          abs(degree.mean()),
                                                                          abs(degree.max())))
    print("distr. of P_i: min={:.2f} mean={:.2f} absmean={:.2f} max={:.2f}".format(np.min(P), np.mean(P),
                                                                                      np.mean(np.abs(P)), np.max(P)))
    print("max. |P|/degree {}".format(np.max(np.abs(P) / np.abs(degree))))
    print("||Y P||_max = {:.2f}".format(score))
    print("-----------------------")

    # determine fixed points
    fps = find_fps(swingpars)
    swingpars.fix_point = fps

    bao.ensure_dir_exists(out_folder)
    np.save(os.path.join(out_folder, "fps.npy"), fps)
    np.save(os.path.join(out_folder, "alphas.npy"), alphas)

    return swingpars

# setups should only run on the master process
@bao.run_on_master
def setup_circle(dynamical_regime, alphas, out_folder, number_of_nodes=100):
    cp = dynamical_regime["consumer/producer"]
    pk = dynamical_regime["P/K"]
    dkh = dynamical_regime["D^2/KH"]

    # Setting H = 1 so that P, K and D are directly the relevant quantities.
    H = 1.

    # scale the coupling distr.
    Y = 1.j * circle_admittance(number_of_nodes, coupling_range=2)
    Y /= 1. * pk * np.mean(np.abs(Y)[Y.nonzero()])

    average_coupling = np.mean(np.abs(Y)[Y.nonzero()])
    min_coupling = np.min(np.abs(Y)[Y.nonzero()])
    max_coupling = np.max(np.abs(Y)[Y.nonzero()])
    degree = np.array([Y[i, i] for i in range(number_of_nodes)])

    # net power
    assert number_of_nodes % (cp + 1) == 0
    n_gen = int(number_of_nodes / (cp + 1))
    Pg = (cp + 1.) / 2.
    Pc = - 1. * Pg / cp
    P = np.ones(number_of_nodes) * Pc
    # regular dispatch
    P[::2] = Pg

    # damping
    D = np.sqrt(dkh) / H

    score = sync_condition(Y, P)

    print("-----------------------")
    print("Set up parameters as:")
    print(r"$\alpha$ ranges from {} to {}".format(alphas[0], alphas[-1]))
    print(
        r"${}\phi'' + {:.2f}\phi' + \sum_j ({} + dK_ij) sin(\phi_i - \phi_j -\alpha) = {:.2f} (prod) / {:.2f} (con)$".format(
            H, D, average_coupling, -cp * Pc, Pc))
    print("distr. of K_ij:    min={:.2f} mean={:.2f} max={:.2f}".format(min_coupling, average_coupling, max_coupling))
    print("distr. of degree:    min={:.2f} mean={:.2f} max={:.2f}".format(abs(degree.min()),
                                                                          abs(degree.mean()),
                                                                          abs(degree.max())))
    print("distr. of P_i:    min={:.2f} mean={:.2f} absmean={:.2f} max={:.2f}".format(np.min(P), np.mean(P),
                                                                                      np.mean(np.abs(P)), np.max(P)))
    print("max. |P|/degree {}".format(np.max(np.abs(P) / np.abs(degree))))
    print("||Y P||_max = {:.2f}".format(score))
    print("-----------------------")

    swingpars = SwingParameters(number_of_nodes)
    swingpars.topology = "circle"
    swingpars.phase_coupling = scipy.sparse.csr_matrix(Y.todense())
    swingpars.damping_coupling = D
    swingpars.input_power = P
    swingpars.alphas = alphas

    # determine fixed points
    fps = find_fps(swingpars)
    swingpars.fix_point = fps

    bao.ensure_dir_exists(out_folder)
    np.save(os.path.join(out_folder, "fps.npy"), fps)
    np.save(os.path.join(out_folder, "alphas.npy"), alphas)

    return swingpars

# setups should only run on the master process
@bao.run_on_master
def setup_northern(dynamical_regime, alphas, out_folder, predefined_dispatch=True):
    cp = dynamical_regime["consumer/producer"]
    pk = dynamical_regime["P/K"]
    dkh = dynamical_regime["D^2/KH"]

    # Setting H = 1 so that P, K and D are directly the relevant quantities.
    H = 1.

    # scale the coupling distr.
    Y = 1.j * northern_admittance(os.path.join("input_data", "northern.json"))
    number_of_nodes = Y.shape[0]
    Y /= 1. * pk * np.mean(np.abs(Y)[Y.nonzero()])

    average_coupling = np.mean(np.abs(Y)[Y.nonzero()])
    min_coupling = np.min(np.abs(Y)[Y.nonzero()])
    max_coupling = np.max(np.abs(Y)[Y.nonzero()])
    degree = np.array([Y[i, i] for i in range(number_of_nodes)])

    # net power
    assert number_of_nodes % (cp + 1) == 0
    n_gen = int(number_of_nodes / (cp + 1))
    Pg = (cp + 1.) / 2.
    Pc = - 1. * Pg / cp

    if predefined_dispatch:
        import pandas
        df = pandas.read_csv(os.path.join("input_data", "northern_dispatch.csv"))
        P = np.array(df[str(cp)])
    else:
        P = np.ones(number_of_nodes) * Pc
        # random dispatch, draw generator location proportional to degree
        idx = np.random.choice(range(number_of_nodes), n_gen, replace=False, p=np.abs(degree)/np.sum(np.abs(degree)))
        P[idx] = Pg

    # damping
    D = np.sqrt(dkh) / H

    swingpars = SwingParameters(number_of_nodes)
    swingpars.topology = "northern"
    swingpars.phase_coupling = scipy.sparse.csr_matrix(Y.todense())
    swingpars.damping_coupling = D
    swingpars.input_power = P
    swingpars.alphas = alphas

    # if no predefined dispatch was loaded, optimise synchronisability
    if not predefined_dispatch:
        P_opt = improve_sync(Y, P, max_trials=swingpars.system_size, flow_limit=1.)
        swingpars.input_power = P_opt
        fp = get_generalized_fixpoint(swingpars)

        swingpars_alpha = copy.deepcopy(swingpars)
        swingpars_alpha.phase_coupling *= np.exp(1.j * alphas[-1])
        fp_alpha = get_generalized_fixpoint(swingpars_alpha)

        # in case no fixed point is found, repeat the process
        trials = 0
        while fp is None or fp_alpha is None:
            print("Trial {} to find working dispatch in northern grid.".format(trials), file=sys.stderr)
            P_opt = improve_sync(Y, P, max_trials=50, flow_limit=1.)

            swingpars.input_power = P_opt
            fp = get_generalized_fixpoint(swingpars)
            swingpars_alpha = copy.deepcopy(swingpars)
            swingpars_alpha.phase_coupling *= np.exp(1.j * alphas[-1])
            fp_alpha = get_generalized_fixpoint(swingpars_alpha)

            trials += 1
            if trials > 10:
                break
        P = P_opt

    score = sync_condition(Y, P)

    print("-----------------------")
    print("Set up parameters as:")
    print(r"$\alpha$ ranges from {} to {}".format(alphas[0], alphas[-1]))
    print(
        r"${}\phi'' + {:.2f}\phi' + \sum_j ({} + dK_ij) sin(\phi_i - \phi_j -\alpha) = {:.2f} (prod) / {:.2f} (con)$".format(
            H, D, average_coupling, -cp * Pc, Pc))
    print("distr. of K_ij:    min={:.2f} mean={:.2f} max={:.2f}".format(min_coupling, average_coupling, max_coupling))
    print("distr. of degree:    min={:.2f} mean={:.2f} max={:.2f}".format(abs(degree.min()),
                                                                          abs(degree.mean()),
                                                                          abs(degree.max())))
    print("distr. of P_i:    min={:.2f} mean={:.2f} absmean={:.2f} max={:.2f}".format(np.min(P), np.mean(P),
                                                                                      np.mean(np.abs(P)), np.max(P)))
    print("max. |P|/degree {}".format(np.max(np.abs(P) / np.abs(degree))))
    print("||Y P||_max = {:.2f}".format(score))
    print("-----------------------")

    # determine fixed points
    fps = find_fps(swingpars)
    swingpars.fix_point = fps

    bao.ensure_dir_exists(out_folder)
    np.save(os.path.join(out_folder, "fps.npy"), fps)
    np.save(os.path.join(out_folder, "alphas.npy"), alphas)

    return swingpars

def get_generalized_fixpoint(swingpar, guess=None, maxiter=int(1e3), post_bif_integrate=None):
    assert isinstance(swingpar, SwingParameters)

    if post_bif_integrate is not None:
        from scipy.integrate import odeint

    def rootfunc(y):
        w_global = y[0]
        phase = np.exp(1.j * y)
        phase[0] = 1.
        flow = swingpar.phase_coupling.dot(phase).conjugate()
        return swingpar.input_power - swingpar.damping_coupling * w_global - swingpar.coupling_factor * np.real(
            phase * flow)

    # If no guess is given we solve the linearized power flow of the lossless system and use the explicit quantity of
    # the equilibrium frequency of lossless systems.
    if guess is None:
        DC_angles = np.linalg.lstsq(np.imag(swingpar.phase_coupling.todense()), swingpar.input_power, rcond=None)[0]
        global_omega = np.sum(swingpar.input_power) / np.sum(swingpar.damping_coupling)
        guess = np.append(DC_angles, global_omega * np.ones_like(DC_angles))

    g2 = np.zeros(swingpar.system_size)

    g2[1:] = guess[1:swingpar.system_size] - guess[0]
    g2[0] = np.average(guess[swingpar.system_size:])

    from scipy.optimize import root

    fixpoint = np.zeros(swingpar.system_dimension)

    res = root(rootfunc, g2, method='krylov', tol=1e-7, options={"disp": False, "maxiter": maxiter})

    if res.success:
        fixpoint[1:swingpar.system_size] = np.mod(res.x[1:swingpar.system_size] + np.pi, 2 * np.pi) - np.pi
        fixpoint[swingpar.system_size:] = res.x[0]

        if post_bif_integrate is not None:
            rhs = gen_swing_rhs(swingpar, custom_dispatch=True)
            state = odeint(rhs,
                           fixpoint + 1e-6 * np.random.randn(swingpar.system_dimension),
                           np.linspace(0, post_bif_integrate, int(post_bif_integrate)),
                           args=(swingpar.input_power, swingpar.phase_coupling,),
                           mxstep=int(1e9))
            fixpoint = state[-1]

        return fixpoint
    else:
        return None

def max_abs_diff(arr, n, adjacency):
    return np.max([np.abs(arr[i] - arr[j]) for (i, j) in zip(*adjacency.nonzero())])

def sync_condition(Y, P):
    """
    sync condition Eqn. 2 in arxiv:1208.0045.
    :param Y:
    :param P:
    :return:
    """
    DC_angles = np.linalg.lstsq(np.imag(Y.todense()), P, rcond=None)[0]
    return max_abs_diff(DC_angles, Y.shape[0], Y)

def improve_sync(Y_init, P_init, max_trials=None, flow_limit=1.):
    """
    Acts in-place. Shuffle the dispatch to find a stable fixed point.
    Optimises sync condition Eqn. 2 in arxiv:1208.0045.
    :param swingpar:
    :return:
    """

    if max_trials is None:
        max_trials = 20.

    start = sync_condition(Y_init, P_init)

    best = copy.deepcopy(start)
    dispatch = copy.deepcopy(P_init)

    for _ in range(int(max_trials)):
        P = np.random.permutation(P_init)
        new_score = sync_condition(Y_init, P)
        if new_score < best:  # and not np.isclose(new_score, 0.):
            dispatch = copy.deepcopy(P)
            best = new_score

    if np.isfinite(best) and best < flow_limit:
        print("Reshuffling improved sync by {:.0f} % from {:.2f} to {:.2f}.".format(100. * (1. - best / start), start, best), file=sys.stderr)
        return dispatch
    else:
        print("No alternative dispatch found. Leave values unchanged. {}".format(start), file=sys.stderr)
        return P_init

def find_fps(swingpar):
    """
    Acts in-place.
    :param swingpar:
    :return:
    """

    fps = np.nan * np.ones((len(swingpar.alphas), swingpar.system_dimension))

    if swingpar.fix_point is None:
        # this is only called when improve_dispatch() has not been executed before
        fps[0, :] = get_generalized_fixpoint(swingpar)
    else:
        fps[0, :] = swingpar.fix_point

    if fps[0, :] is None:
        raise ValueError("Couldn't determine lossless fixed point.")

    for i, alpha in enumerate(swingpar.alphas[1:]):
        swingpar_alpha = copy.deepcopy(swingpar)
        swingpar_alpha.phase_coupling = swingpar_alpha.phase_coupling * np.exp(1.j * alpha)
        res = get_generalized_fixpoint(swingpar_alpha, guess=fps[i, :])
        if res is not None:
            fps[i+1, :] = res

    return fps

