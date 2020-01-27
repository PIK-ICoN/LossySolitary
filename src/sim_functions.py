import copy
import os
from functools import reduce

import baobap as bao
import h5py
import numpy as np
import scipy.sparse
from sklearn.cluster import DBSCAN


# ------------------------------------------------------------------------------------------------------------------- #
# baobap setup

class SwingParameters():
    """
        Data type for the simulation parameters. This is the data passed to define the right hand side of the ODE.
    """

    def __init__(self, system_size=1):
        """
        :param system_size: the size of the system
        :return:
        """
        self.topology = ""
        self.system_size = system_size
        self.system_dimension = 2 * system_size
        self.phase_coupling = np.zeros((system_size, system_size), dtype=np.float64)
        self.damping_coupling = np.zeros(system_size, dtype=np.float64)
        self.input_power = np.zeros(system_size, dtype=np.float64)
        self.fix_point = None
        self.coupling_factor = 1.
        self.alphas = 0.

def load_fps_from_file(fps_folder, batch, run):
    #bao.pprint("loading fixed points from: ", fps_folder)

    with h5py.File(os.path.join(fps_folder, "fps_{}.hdf".format(batch)), mode='r') as h5f:
        phase_coupling = scipy.sparse.csr_matrix(h5f["Y"])
        input_power = np.array(h5f["P"])
        fix_point = np.array(h5f["fp"])[run]

    return phase_coupling, input_power, fix_point

def save_fps_to_file(swingpar, fps_dir, batch):
    bao.ensure_dir_exists(fps_dir)
    #bao.pprint("saving fixed points to: ", fps_dir)

    with h5py.File(os.path.join(fps_dir, "fps_{}.hdf".format(batch)), mode='w') as h5f:
        h5f["Y"] = np.abs(swingpar.phase_coupling.todense())
        h5f["P"] = swingpar.input_power
        h5f["fp"] = swingpar.fix_point
        h5f["alpha"] = swingpar.alphas

def update_rc_from_ensemble(fps_folder, swingpars, batch, run, sample_size=100):
    """
    Y, P, fps should be generated only once per batch. Hence the values are saved to a file in fps_folder.
    If it already exists, the values are read in.
    :param fps_folder: 
    :param swingpars: 
    :param batch: 
    :param sample_size: 
    :return: 
    """

    # For each batch, the first run saves the fixed points to a file, consecutive runs load the data from the file.
    # This should be save as each batch operates on its own file.
    if os.path.exists(os.path.join(fps_folder, "fps_{}.hdf".format(batch))):
        return load_fps_from_file(fps_folder, batch, run)
    else:
        from src.parameters_and_fixed_points import determine_synchronizable_grid, find_fps
        # create a fresh copy of swingparameters
        sp = copy.deepcopy(swingpars)
        sp.fix_point = None
        # redraw until fixed point was found
        tries = 100
        for i in range(tries):
            try:
                pk = np.abs(sp.input_power).mean() / np.mean(np.abs(sp.phase_coupling)[sp.phase_coupling.nonzero()])
                Y, P, _ = determine_synchronizable_grid(sp.input_power, pk, sp.system_size, sample_size=sample_size)
                sp.phase_coupling = Y
                sp.input_power = P
                fps = find_fps(sp)
                sp.fix_point = fps
                save_fps_to_file(sp, fps_folder, batch)
            except:
                if i < tries - 1:
                    continue
                else:
                    raise ValueError("Max. number of tries.")
            break
        return Y, P, fps[run]


@bao.run_on_master
def find_sim_dir(base_dir, base_str):
    cnum = 0
    while os.path.exists(os.path.join(base_dir, base_str + str(cnum))):
        cnum += 1
    sim_dir = os.path.join(base_dir, base_str + str(cnum))
    return sim_dir


@bao.run_on_master
def write_swingpars_to_resfile(result_file, swingpar):
    import h5py
    with h5py.File(result_file) as h5f:
        swing_for_save = copy.deepcopy(swingpar)
        swing_for_save.phase_coupling = swingpar.phase_coupling.todense()
        h5f.create_group("Swing_Parameters")
        bao.save_class_to_hdf5_group(swing_for_save, h5f["Swing_Parameters"])


def gen_swing_rhs(swing_parameters, custom_dispatch=False):
    """
    This function defines a rhs function with the data given by simulation_run using sparse matrix multiplication.
    :param swing_parameters: an instance of SimulationRun that defines the system parameters.
    :return: A function of signature rhs(y, t)
    """
    assert isinstance(swing_parameters, SwingParameters)

    size_of_system = int(swing_parameters.system_size)
    input_power = np.array(swing_parameters.input_power, dtype=np.float64)
    damping_coupling = np.array(swing_parameters.damping_coupling, dtype=np.float64)
    phase_coupling_sp = scipy.sparse.csr_matrix(swing_parameters.phase_coupling, dtype=np.complex128)
    coupling_factor = swing_parameters.coupling_factor  # global coupling prefactor

    if custom_dispatch:
        def right_hand_side_sparse(y, _, e_mi_alpha, ensemble_p, ensemble_y):
            phases = np.exp(1.j * y[:size_of_system])
            return np.append(y[size_of_system:], ensemble_p - damping_coupling * y[size_of_system:] -
                             coupling_factor * np.real(phases * ensemble_y.dot(e_mi_alpha * phases).conjugate()))
    else:
        # We start with phase coupling purely positive imaginary, and then rotate it towards a positive real component by
        # multiplying e^{-i \alpha} on.
        def right_hand_side_sparse(y, _, e_mi_alpha):
            phases = np.exp(1.j * y[:size_of_system])
            return np.append(y[size_of_system:], input_power - damping_coupling * y[size_of_system:] -
                             coupling_factor * np.real(phases * phase_coupling_sp.dot(e_mi_alpha * phases).conjugate()))

    return right_hand_side_sparse


def def_rc_gen(swingpar, fps, alphas, freq_var=10., phase_var=np.pi, fps_folder=None):
    assert isinstance(swingpar, SwingParameters)

    if swingpar.topology == "ensemble":
        def generate_run_conditions(batch, run):
            phase_coupling, input_power, ic = update_rc_from_ensemble(fps_folder, swingpar, batch, run, sample_size=100)
            # The node to disturb:
            n = np.random.randint(swingpar.system_size)
            # Phase and frequency perturbation, with variance given as above:
            ic[n] = (2. * np.random.rand() - 1.) * phase_var
            ic[n + swingpar.system_size] = (2. * np.random.rand() - 1.) * freq_var
            return ic, (np.exp(1.j * alphas[run]), input_power, 1.j * phase_coupling,)
    else:
        def generate_run_conditions(batch, run):
            # The node to disturb:
            ic = np.copy(fps[run])
            n = np.random.randint(swingpar.system_size)
            # Phase and frequency perturbation, with variance given as above:
            ic[n] = (2. * np.random.rand() - 1.) * phase_var
            ic[n + swingpar.system_size] = (2. * np.random.rand() - 1.) * freq_var
            e_alpha = np.exp(1.j * alphas[run])
            return ic, (e_alpha,)

    return generate_run_conditions


def def_rc_gen_global(swingpar, fps, alphas, freq_var=10., phase_var=np.pi):
    assert isinstance(swingpar, SwingParameters)

    def generate_run_conditions(batch, run):
        # The node to disturb:

        ic = np.copy(fps[run])

        # n = np.random.randint(swingpar.system_size)

        # Phase and frequency perturbation, with variance given as above:
        ic[:swingpar.system_size] += (2. * np.random.rand(swingpar.system_size) - 1.) * phase_var
        ic[swingpar.system_size:] += (2. * np.random.rand(swingpar.system_size) - 1.) * freq_var

        e_alpha = np.exp(1.j * alphas[run])

        return ic, (e_alpha,)

    return generate_run_conditions


def def_rc_gen_square(swingpar, n_phase, n_freq, fps, alphas, max_freq=10., max_phase=np.pi, node=0):
    assert isinstance(swingpar, SwingParameters)

    n = node

    def generate_run_conditions(batch, run):
        # The node to disturb:

        ic = np.copy(fps[run])

        # Phase and frequency perturbation, with variance given as above:
        ic[n] = max_phase - (batch % n_phase) / (n_phase - 1) * 2. * max_phase
        ic[n + swingpar.system_size] = max_freq - np.floor(batch / n_phase) / (n_freq - 1) * 2. * max_freq

        e_alpha = np.exp(1.j * alphas[run])

        return ic, (e_alpha,)

    return generate_run_conditions


def def_swing_ob(swingpar, times):
    n_times = len(times) - len(times) // 10
    frequ_id = list(range(swingpar.system_size, 2 * swingpar.system_size))

    def swing_ob(time_series, rc):
        return {"asymptotic_frequencies": np.average(time_series[n_times:, frequ_id], axis=0)}

    return swing_ob


def def_ftbs_ob(swingpar, times, threshold=None, freq_var=10., phase_var=np.pi / 2):
    frequ_id = list(range(swingpar.system_size, 2 * swingpar.system_size))

    R = np.append(np.tile(phase_var, swingpar.system_size), np.tile(freq_var, swingpar.system_size))

    stepfunc = np.vectorize(lambda x, f: 0. if x < f else 1.)

    prod = lambda a, b: a * b

    if threshold is None:
        threshold = np.logspace(-6, -1, 6)

    def iota_uniform(point=[0, 0], rel=True):
        # px, py = np.abs(point)
        # a, b = R
        if rel:
            point = np.array(point) - swingpar.fix_point

        overlap = reduce(prod, stepfunc(2. * R - np.array(point), 0))

        if overlap == 0:
            return 2.
        else:
            # V grows exponentially with the system size
            V = np.sum(np.log10(2. * R))
            return 2. - 2. * np.exp(np.log10(overlap) - V)
        # return 2. - 2. * max(0, 2. * a - point[0]) * max(0, 2. * b - point[1]) / V

    def find_idx(vals, e, first=False, less=False, nothing=[]):
        if less:
            idx = np.where(np.array(vals) < e)[0]
        else:
            idx = np.where(np.array(vals) > e)[0]
        # print idx
        if len(idx) is 0:
            return nothing
        else:
            if first:
                return idx[0]
            else:
                return idx[-1]

    def ftbs_ob(time_series, rc):
        val = [iota_uniform(p) for p in time_series]
        return {"epsilon": threshold, "return time": np.array([find_idx(val, t, nothing=np.nan) for t in threshold])}

    return ftbs_ob

# plots should only be created on the master process
@bao.run_on_master
def plot_fps(sim_dir, swingpar):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = swingpar.alphas
    fps = swingpar.fix_point

    fig_freq = plt.figure()
    plt.title("Frequencies of the sync state")
    plt.xlabel("Phase shift")
    plt.plot(alphas, fps[:, swingpar.system_size:])
    plt.savefig(os.path.join(sim_dir, "sync_frequency.png"))

    plt.close(fig_freq)

    phase_change = np.abs(fps[-1, :swingpar.system_size] - fps[0, :swingpar.system_size])

    rgba_colors = np.zeros((swingpar.system_size, 4))
    rgba_colors[:, 3] = 0.01 + 0.6 * phase_change / np.max(phase_change)

    fig_phase = plt.figure()
    plt.title("Phases of the sync state\n(strongly changing phases highlighted)")
    plt.xlabel("Phase shift")
    for fp, c in zip(fps.T[:swingpar.system_size], rgba_colors):
        plt.plot(alphas, fp * 180. / np.pi, color=c)
    plt.ylabel("angle °")
    plt.savefig(os.path.join(sim_dir, "sync_phases.png"))

    plt.close(fig_phase)

# plots should only be created on the master process
@bao.run_on_master
def analyse_ts(result_file):
    ts_dir = os.path.join(os.path.dirname(result_file), "ts_dir")

    import matplotlib
    from matplotlib import pyplot
    matplotlib.use("Agg")
    import awesomeplot.core as ap

    class NewPlot(ap.Plot):
        def space_time_plot(self, x, y, z, labels=['x', 'y', 'z'],
                            zmin=0, zmax=1, colorbar=True, sym=False, horizontal=False, pi=None):

            assert len(labels) == 3

            # Issue warning if z contains NaN or Inf
            if sym:
                cmap = pyplot.get_cmap("sym")
                cmap.set_under(cmap(0))
                cmap.set_over(cmap(255))
                extend = "both"
            else:
                cmap = pyplot.get_cmap("YlOrBr")
                cmap.set_over(cmap(255))
                extend = "max"

            pyplot.gca().patch.set_color('#8e908f')  # print the Nan/inf Values in black)

            fig, ax = pyplot.subplots(nrows=1, ncols=1)

            fig.tight_layout()

            c = ax.imshow(z, extent=[x.min(), x.max(), y.min(), y.max()],
                          interpolation="none", aspect="auto",
                          cmap=cmap, origin='lower', vmin=zmin, vmax=zmax)
            c.set_clim([zmin, zmax])

            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])

            if pi == "xaxis":
                x_label = np.empty(np.size(ax.get_xticks()), dtype='object')
                for i in range(np.size(ax.get_xticks())):
                    x_label[i] = str(ax.get_xticks()[i]) + "$\pi$"
                ax.set_xticklabels(x_label)

            if pi == "yaxis":
                y_label = np.empty(np.size(ax.get_yticks()), dtype='object')
                for i in range(np.size(ax.get_yticks())):
                    y_label[i] = str(ax.get_yticks()[i]) + "$\pi$"
                ax.set_yticklabels(y_label)

            if colorbar and horizontal:
                fig.colorbar(c, label=labels[2], orientation='horizontal', pad=0.2, shrink=0.8, extend=extend)
            elif colorbar:
                fig.colorbar(c, label=labels[2], shrink=0.8,
                             extend=extend)  # not so cool for smalll numbers format=r"%.1f"

            self.figures.append(fig)

            return fig

        def snapshot(self, y1, y2, labels=['x', 'y1', 'y2']):

            assert len(labels) == 3

            n = max(len(y1), len(y2))

            x = list(range(n))

            fig, ax = pyplot.subplots(nrows=2, ncols=1, sharex=True)

            # determine boundaries and scale

            xmin = 0
            xmax = n - 1
            xmargin = 1. * n / 200.
            scale = np.log(1 + 1. * n / len(x))

            ax[0].plot(x, y1, linewidth=0, marker=".", ms=10. * scale, label=labels[1])
            ax[0].hlines(y=np.pi, xmin=0, xmax=n - 1, linewidth=1)
            ax[1].plot(x, y2, linewidth=0, marker=".", ms=10. * scale, label=labels[2])
            ax[1].hlines(y=0, xmin=0, xmax=n - 1, linewidth=1)

            y_label = np.empty(np.size(ax[0].get_yticks()), dtype='object')
            for i in range(np.size(ax[0].get_yticks())):
                y_label[i] = str(round(ax[0].get_yticks()[i] / np.pi, 1)) + "$\pi$"
            ax[0].set_yticklabels(y_label)

            ax[1].set_xlabel(labels[0])
            ax[0].set_ylabel(labels[1])
            ax[1].set_ylabel(labels[2])

            fig.tight_layout()

            self.figures.append(fig)

            return fig

        def polar_plot(self, op, label="r", grid=False):
            fig = pyplot.figure()
            ax = pyplot.subplot(111, polar=True)
            ax.spines['polar'].set_visible(grid)
            ax.plot(np.angle(op), np.abs(op), "-", marker="o", zorder=3, alpha=0.2)

            xT = pyplot.xticks()[0]
            xL = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$', r'$\frac{5\pi}{4}$',
                  r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
            pyplot.xticks(xT, xL)

            ax.set_xlabel(label)
            fig.tight_layout()
            self.figures.append(fig)
            return fig

    canvas = NewPlot(output="paper")
    canvas.set_default_colours("discrete")
    canvas.figure_format = "png"

    with bao.h5py.File(result_file, mode='r') as h5f:
        times = h5f['times'][()]

    counter = 0
    for fn in os.listdir(ts_dir):
        if counter > 9:
            break
        print(fn)
        states = np.load(os.path.join(ts_dir, fn))["arr_0"]
        n = int(states.shape[1] / 2.)
        freq = states[:, n:]
        mfreq = np.mean(states[-int(len(times) * 0.3):, n:], axis=0)
        maxf = np.abs(states[-int(len(times) * 0.3):, n:] - mfreq).max()
        frequency_time_plot = canvas.space_time_plot(x=np.arange(n), y=times, z=np.abs(freq - mfreq),
                                                     labels=["node ID", "time", r"$\vert\omega_k\vert$"],
                                                     colorbar=True, sym=False, horizontal=False, zmin=0, zmax=maxf
                                                     )
        canvas.save(os.path.join(ts_dir, str(fn) + "_flame"), frequency_time_plot)
        snapshot = canvas.snapshot(np.mod(states[-1, :n], 2 * np.pi), states[-1, n:],
                                   labels=["node ID", r"phase $\phi_k$", r"frequency $\omega_k$"])
        canvas.save(os.path.join(ts_dir, str(fn) + "_snap"), snapshot)
        op = np.mean(np.exp(1.j * states[:, :n]), axis=1)
        op_r = canvas.polar_plot(op, label="")
        canvas.save(os.path.join(ts_dir, str(fn) + "_op"), op_r)
        counter += 1

    del canvas


def detect_n_desync(freq):
    # How many points are outside the largest cluster of asymptotic frequencies:
    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=0.3, min_samples=1).fit(freq[:, np.newaxis])

    labels = db.labels_

    s_labels = set(labels)

    n_clusters = len(set(labels))

    size_of_cluster = np.zeros(n_clusters)

    for i, l in enumerate(s_labels):
        size_of_cluster[i] = np.count_nonzero(labels == l)

    return int(len(freq) - np.max(size_of_cluster))


def detect_solitary(freq):
    # Return the number of points outside the largest cluster, their location and their average frequency
    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=0.3, min_samples=1).fit(freq[:, np.newaxis])

    labels = db.labels_

    s_labels = set(labels)

    n_clusters = len(set(labels))

    size_of_cluster = np.zeros(n_clusters)

    max_size = 0

    for i, l in enumerate(s_labels):
        size_of_cluster[i] = np.count_nonzero(labels == l)
        if size_of_cluster[i] > max_size:
            max_size = size_of_cluster[i]
            l_max = l

    n_desync = int(len(freq) - max_size)

    idxs = (labels != l_max).nonzero()[0]

    average_frequency = np.mean(freq[idxs])  # Might be nan!

    return n_desync, idxs, average_frequency


def detect_clusters(freq):
    # DBSCAN requires a 2-d array of features. our frequencies are typically just one number, so we add an axis.
    if len(freq.shape) == 1:
        X = freq[:, np.newaxis]
    else:
        X = freq

    db = DBSCAN(min_samples=1).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
    n_clusters = len(unique_labels)
    cluster_means = np.zeros(n_clusters)
    for i, k in enumerate(unique_labels):
        cluster_means[i] = np.nanmean(X[labels == k, 0])

    sol_idx = np.nan
    if n_clusters == 2 and 1 in cluster_sizes:
        sol_idx = int(np.where(labels == unique_labels[cluster_sizes == 1][0])[0][0])

    return n_clusters, cluster_sizes, cluster_means, sol_idx