import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as pyplot
import networkx as nx
import numpy as np
import warnings
import src.awesomeplot as ap
import baobap as bao
import os, gc

from src import detect_clusters


class NewPlot(ap.Plot):
    def space_time_plot(self, x, y, z, labels=['x', 'y', 'z'],
                        zmin=0, zmax=1, colorbar=True, sym=False, horizontal=False, pi=None):

        assert len(labels) == 3

        # Issue warning if z contains NaN or Inf
        if not np.isfinite(z).all():
            warnings.warn("Since z is not finite, it would be better to use layout=False.")

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
            fig.colorbar(c, label=labels[2], shrink=0.8, extend=extend)  # not so cool for smalll numbers format=r"%.1f"

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


@bao.run_on_master
def plot_full_results(sim_dir, run=["asbs", "global", "pp"], suffix=""):

    _, ana = bao.load_state_for_analysis(os.path.join(sim_dir, "analysis.p"))
    sim_dir, alphas, fps, swingpar, node, sides, pp_sparsity = ana

    result_file = os.path.join(sim_dir, "results_ASBS.hdf")
    if "asbs" in run:
        bifurcation_lineplot(result_file, sim_dir, alphas, fps, swingpar, suffix=suffix)

    gc.collect()

    result_file = os.path.join(sim_dir, "results_globalBS.hdf")
    if "global" in run:
        bifurcation_lineplot(result_file, sim_dir, alphas, fps, swingpar, suffix=suffix)

    gc.collect()

    result_file = os.path.join(sim_dir, "results_pp.hdf")
    if "pp" in run:
        pp_picture(result_file, sim_dir, alphas[::pp_sparsity], fps[::pp_sparsity], swingpar, node, sides, suffix=suffix)

    gc.collect()

@bao.run_on_master
def grouped_asbs_plot(dynamical_regimes, folder="resubmission/circle/", suffix=""):
    # initialise plotting canvas
    import matplotlib
    matplotlib.use("Agg")
    import awesomeplot.core as ap
    from cycler import cycler

    # initialise plotting canvas
    canvas = ap.Plot(output="paper")
    canvas.figure_format = "png"

    matplotlib.pyplot.style.use("ggplot")
    matplotlib.rcParams.update({'font.size': 12,
                                'xtick.labelsize': 12,
                                'ytick.labelsize': 12,
                                'axes.linewidth': 4,
                                'figure.dpi': 600,
                                'axes.prop_cycle': cycler(color=["r", "tab:purple", "blue"],
                                                          linestyle=["-", "--", "-."])
                                })


    def dict2list(d, k):
        return np.unique(["{:.3f}".format(dr[k]).rstrip('.0').replace(".", "p") for dr in d]) #:.3f

    cp_list = dict2list(dynamical_regimes, "consumer/producer")
    pk_list = dict2list(dynamical_regimes, "P/K")
    dkh_list = dict2list(dynamical_regimes, "D^2/KH")

    # one figure for each consumer/producer ratio
    for cp in cp_list:

        fig, grid = pyplot.subplots(ncols=len(dkh_list),
                                    nrows=len(pk_list),
                                    sharex="col",
                                    sharey="row",
                                    #constrained_layout=True,
                                    figsize=(11.69, 8.27))


        # loop over dynamical regimes with given cp
        for ax_pk, pk in enumerate(pk_list):
            for ax_dkh, dkh in enumerate(dkh_list):
                name = ("cp_{}_PK_{}_D2KH_{}").format(cp, pk, dkh) + suffix
                sim_dir = os.path.join(folder, name)
                print(name, os.path.exists(sim_dir))


                ax = grid[ax_pk][ax_dkh]
                # skip missing simulations
                try:
                    alphas = np.load(os.path.join(sim_dir, "alphas.npy"))
                    asbs_result_dir = os.path.join(sim_dir, "results_ASBS")
                    number_of_desync = np.load(os.path.join(asbs_result_dir, "number_of_desync.npy"))
                    n_clusters = np.load(os.path.join(asbs_result_dir, "ncluster.npy"))
                    exotic_solitary = np.load(os.path.join(asbs_result_dir, "exotic_solitary.npy"))
                    fraction_of_solitaries = np.mean(number_of_desync == 1, axis=0)
                    fraction_of_exotic_solitaries = np.mean((exotic_solitary == 1) & (number_of_desync == 1), axis=0)
                    fraction_of_sync = np.mean(number_of_desync == 0, axis=0)
                    fraction_of_desync = np.mean(number_of_desync > 0, axis=0)

                    #ax.plot(alphas, fraction_of_exotic_solitaries)
                    #ax.plot(alphas, fraction_of_sync)
                    #ax.plot(alphas, fraction_of_solitaries)
                    canvas.draw_lineplot(figax=(fig, ax),
                                                       x=alphas,
                                                       lines={"phase sync": fraction_of_sync,
                                                              "solitary": fraction_of_solitaries,
                                                              "exotic solitary": fraction_of_exotic_solitaries},
                                                       # "desync":average_number_of_desync},
                                                       labels=(r"phase lag $\alpha$", ""),
                                                       legend=False,
                                                       marker=None,
                                                       infer_layout=False)
                    ax.plot(alphas, fraction_of_desync, "k-", zorder=-1, alpha=0.5)
                    ax.set_ylim(0, 1)
                    ax.set_xlim(0, alphas[-1])
                except:
                    continue

                ax.set_title("P/K={} D^2/KH={}".format(pk[:5], dkh))

        fig.savefig(os.path.join(folder, "asbs_cp_{}.png".format(cp)))
        pyplot.close(fig)
        gc.collect()

@bao.run_on_master
def grouped_pp_plot(dynamical_regimes, folder="resubmission/circle/", suffix="", alpha_idx=-1):
    pyplot.style.use("seaborn-poster")
    cmap = pyplot.get_cmap("Accent")
    bounds = [-1, 0, 1, 2, 3]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    mpl.rcParams.update({'figure.dpi': 600,
                         'font.size': 10,
                         'xtick.labelsize': 10,
                         'ytick.labelsize': 10,
                         'axes.prop_cycle': pyplot.cycler(color=["r", "tab:purple", "blue"],
                                                          linestyle=["-", "--", "-."])
                        })

    def dict2list(d, k):
        return np.unique(["{:.3f}".format(dr[k]).rstrip('.0').replace(".", "p") for dr in d]) #:.3f

    cp_list = dict2list(dynamical_regimes, "consumer/producer")
    pk_list = dict2list(dynamical_regimes, "P/K")
    dkh_list = dict2list(dynamical_regimes, "D^2/KH")

    alphas = [np.nan]

    # one figure for each consumer/producer ratio
    for cp in cp_list:

        fig, grid = pyplot.subplots(ncols=len(dkh_list),
                                    nrows=len(pk_list),
                                    sharex="col",
                                    sharey="row",
                                    #constrained_layout=True,
                                    figsize=(11.69, 8.27))

        # loop over dynamical regimes with given cp
        for ax_pk, pk in enumerate(pk_list):
            for ax_dkh, dkh in enumerate(dkh_list):
                name = ("cp_{}_PK_{}_D2KH_{}").format(cp, pk, dkh).replace(".", "p") + suffix
                sim_dir = os.path.join(folder, name)
                print(name, os.path.exists(sim_dir))

                ax = grid[ax_pk][ax_dkh]
                # skip missing simulations
                try:
                    _, ana = bao.load_state_for_analysis(os.path.join(sim_dir, "analysis.p"))
                    sides = ana[5]
                    swingpar = ana[3]
                    pp_sparsity = ana[6]
                    alphas = np.load(os.path.join(sim_dir, "alphas.npy"))[::pp_sparsity]
                    pp_result_dir = os.path.join(sim_dir, "results_pp")

                    number_of_desync = np.load(os.path.join(pp_result_dir, "number_of_desync.npy"))
                    n_clusters = np.load(os.path.join(pp_result_dir, "ncluster.npy"))

                    im = ax.imshow(number_of_desync[:, alpha_idx].reshape((sides, sides)),
                                   extent=[-np.pi, np.pi, -10, 10], aspect="auto",
                                   cmap=cmap, interpolation="nearest",  # vmin=0, vmax=10,
                                   norm=norm)
                    # ax.set_xlabel(r"$\phi_k$")
                    # ax.set_ylabel(r"$\omega_k$")
                except:
                    continue

                ax.set_title("P/K={} D^2/KH={}".format(pk[:5], dkh))

        fig.savefig(os.path.join(folder, "pp_alpha_{}_cp_{}.png".format(alphas[alpha_idx], cp)))
        pyplot.close(fig)
        gc.collect()

@bao.run_on_master
def clustering_frequency_data(res_dir, result_file, swingpar, overwrite=False):
    bao.ensure_dir_exists(res_dir)

    if not overwrite and \
            os.path.exists(os.path.join(res_dir, "number_of_desync.npy")) and \
            os.path.exists(os.path.join(res_dir, "ncluster.npy")) and \
            os.path.exists(os.path.join(res_dir, "sync.npy")) and \
            os.path.exists(os.path.join(res_dir, "solitary.npy")) and \
            os.path.exists(os.path.join(res_dir, "exotic_solitary.npy")) and \
            os.path.exists(os.path.join(res_dir, "composite.npy")) and \
            os.path.exists(os.path.join(res_dir, "exotic_composite.npy")) and \
            os.path.exists(os.path.join(res_dir, "other.npy")):

        number_of_desync = np.load(os.path.join(res_dir, "number_of_desync.npy"))
        n_clusters = np.load(os.path.join(res_dir, "ncluster.npy"))
        sync = np.load(os.path.join(res_dir, "sync.npy"))
        solitary = np.load(os.path.join(res_dir, "solitary.npy"))
        exotic_solitary = np.load(os.path.join(res_dir, "exotic_solitary.npy"))
        composite = np.load(os.path.join(res_dir, "composite.npy"))
        exotic_composite = np.load(os.path.join(res_dir, "exotic_composite.npy"))
        other = np.load(os.path.join(res_dir, "other.npy"))

        return number_of_desync, n_clusters, sync, solitary, exotic_solitary, composite, exotic_composite, other
    else:
        if overwrite:
            try:
                os.system("rm {}/*.npy".format(res_dir))
            except:
                print(os.path.join(res_dir, "*.npy"))
                print("Could not delete npy files.")

        af = bao.load_field_from_results(result_file, "asymptotic_frequencies")
        powers = swingpar.input_power
        batches, runs, sys_size = af.shape

        number_of_desync = np.zeros((batches, runs))
        n_clusters = np.zeros((batches, runs))

        # We want six different categories:
        ## 0 - sync: no desynchronised nodes, one coherent cluster
        sync = np.full((batches, runs), False)
        solitary = np.full((batches, runs), False)
        exotic_solitary = np.full((batches, runs), False)
        composite = np.full((batches, runs), False)
        exotic_composite = np.full((batches, runs), False)
        other = np.full((batches, runs), False)



        for b in range(batches):
             for r in range(runs):
                # need to catch NaN's or Inf
                if not np.all(np.isfinite(af[b, r])):
                    n_clusters[b, r] = np.nan
                    exotic_solitary[b, r] = np.nan
                    number_of_desync[b, r] = np.nan
                else:
                    # DBSCAN of average phase velocity
                    n_clusters[b, r], c_sizes, c_means, idx = detect_clusters(af[b, r])
                    # count nodes outside largest cluster
                    number_of_desync[b, r] = sys_size - np.max(c_sizes)

                    # no clusters -> synchronisation
                    if number_of_desync[b, r] == 0:
                        sync[b, r] = True
                    # there is a single desynchronised node -> a solitary
                    elif number_of_desync[b, r] == 1:
                        avg_pv = np.sign(af[b, r, idx])
                        nat_freq = np.sign(powers[idx])
                        # record whether solitary is exotic or not
                        exotic_solitary[b, r] = True if avg_pv * nat_freq == -1. else False
                        solitary[b, r] = not exotic_solitary[b, r]
                    # multiple desynchronised nodes or clusters
                    elif (number_of_desync[b, r] > 1):
                        if 1 in c_sizes: # at least one solitary is present
                            if np.any(np.sign(af[b, r, idx]) * np.sign(powers[idx]) == -1):
                                # there is at least one exotic solitary
                                exotic_composite[b, r] = True
                            else:
                                # no exotic solitary is present
                                composite[b, r] = True
                        else:
                            # cluster sync and other states
                            other[b, r] = True

        np.save(os.path.join(res_dir, "other"), other)
        np.save(os.path.join(res_dir, "composite"), composite)
        np.save(os.path.join(res_dir, "exotic_composite"), exotic_composite)
        np.save(os.path.join(res_dir, "exotic_solitary"), exotic_solitary)
        np.save(os.path.join(res_dir, "solitary"), solitary)
        np.save(os.path.join(res_dir, "sync"), sync)
        np.save(os.path.join(res_dir, "number_of_desync"), number_of_desync)
        np.save(os.path.join(res_dir, "ncluster"), n_clusters)

        return number_of_desync, n_clusters, sync, solitary, exotic_solitary, composite, exotic_composite, other



@bao.run_on_master
def bifurcation_lineplot(result_file, sim_dir, alphas, fps, swingpar, suffix=""):

    head, tail = os.path.split(result_file)
    res_dir = os.path.join(head, tail.split('.')[0] + suffix)
    bao.ensure_dir_exists(res_dir)

    number_of_desync, n_clusters, exotic_solitary = clustering_frequency_data(res_dir, result_file, swingpar)
    average_number_of_desync = np.nanmean(number_of_desync, axis=0)
    fraction_of_solitaries = np.nanmean(number_of_desync == 1, axis=0)
    fraction_of_exotic_solitaries = np.nanmean((exotic_solitary == 1) & (number_of_desync == 1), axis=0)
    fraction_of_sync = np.nanmean(number_of_desync == 0, axis=0)


    import matplotlib
    matplotlib.use("Agg")


    # plot average number of desynchronized oscillators vs. \alpha
    fig_n_desync, ax = pyplot.subplots(nrows=1, ncols=1)

    x = np.tile(alphas, number_of_desync.shape[0])
    y = number_of_desync.flatten()
    pp = ax.hist2d(x, y, bins=(len(alphas), swingpar.system_size-1), normed=True, cmap=pyplot.cm.get_cmap("magma_r"),
                   range=[[alphas.min(), alphas.max()], [np.nanmin(y), np.nanmax(y)]])
    ax.plot(alphas, average_number_of_desync, "k:")
    fig_n_desync.colorbar(pp[3], ax=ax)
    ax.set_ylim([1, swingpar.system_size])
    ax.set_yscale("log")
    ax.set_xlabel(r"phase lag $\alpha$")
    ax.set_ylabel("number of desync oscillators")

    # fig_n_desync = canvas.draw_lineplot(x=alphas,
    #                                    lines={"desync": average_number_of_desync},  #"desync":average_number_of_desync},
    #                                    labels=(r"phase lag $\alpha$", ""),
    #                                    legend=False,
    #                                    marker=None, infer_layout=False)

    fig_n_desync.set_size_inches(5, 4)
    fig_n_desync.savefig(os.path.join(res_dir, "n_desync.pdf"), bbox_inches='tight')

    # plot average number of sync clusters vs. \alpha

    fig_n, ax = pyplot.subplots(nrows=1, ncols=1)

    x = np.tile(alphas, n_clusters.shape[0])
    y = n_clusters.flatten()
    pp = ax.hist2d(x, y, bins=(len(alphas), np.nanmax(y)), normed=True, cmap=pyplot.cm.get_cmap("magma_r"),
                   range=[[alphas.min(), alphas.max()], [np.nanmin(y), np.nanmax(y)]])
    ax.plot(alphas, np.nanmean(n_clusters, axis=0), "k:")
    fig_n.colorbar(pp[3], ax=ax)
    #ax.set_ylim([0, swingpar.system_size])
    ax.set_xlabel(r"phase lag $\alpha$")
    ax.set_ylabel("number of clusters")

    # fig_n = canvas.draw_lineplot(x=alphas,
    #                              lines={"mean": np.mean(n_clusters, axis=0), "max": np.max(n_clusters, axis=0)},
    #                              labels=[r"phase lag $\alpha$", ""],
    #                              legend=False,
    #                              marker=None, infer_layout=False)
    fig_n.set_size_inches(5, 4)
    fig_n.savefig(os.path.join(res_dir, "n_clust.pdf"), bbox_inches='tight')

    import awesomeplot.core as ap
    from cycler import cycler

    # initialise plotting canvas
    canvas = ap.Plot(output="paper")
    canvas.figure_format = "pdf"

    # canvas.save(os.path.join(res_dir, "n_desync"), fig_n_desync)


    matplotlib.pyplot.style.use("ggplot")
    matplotlib.rcParams.update({'font.size': 24,
                                'xtick.labelsize': 20,
                                'ytick.labelsize': 20,
                                'figure.figsize': (8, 8),
                                'axes.linewidth': 4,
                                'figure.dpi': 600,
                                'axes.prop_cycle': cycler(color=["r", "tab:purple", "blue"],
                                                          linestyle=["-", "--", "-."])
                                })

    # plot ASBS vs. \alpha
    fig_combine = canvas.draw_lineplot(x=alphas,
                                       lines={"phase sync": fraction_of_sync, "solitary": fraction_of_solitaries,
                                              "exotic solitary": fraction_of_exotic_solitaries},
                                       # "desync":average_number_of_desync},
                                       labels=(r"phase lag $\alpha$", ""),
                                       legend=False,
                                       marker=None,
                                       infer_layout=False)
    fig_combine.axes[0].set_ylim(0, 1)
    fig_combine.set_size_inches(5, 4)
    canvas.save(os.path.join(res_dir, "bs_sol"), fig_combine)

    del canvas



@bao.run_on_master
def pp_picture(result_file, sim_dir, alphas, fps, swingpar, node, sides, maxi=[np.pi, 10.], suffix=""):

    res_dir = os.path.join(sim_dir, "results_pp/")

    bao.ensure_dir_exists(res_dir)

    number_of_desync, n_clusters, exotic_solitary = clustering_frequency_data(res_dir, result_file, swingpar)

    x = np.linspace(-maxi[0], maxi[0], sides, endpoint=True)
    y = np.linspace(-maxi[1], maxi[1], sides, endpoint=True)


    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update({'font.size': 22, 'legend.fontsize':16})

    plt.style.use("seaborn-poster")

    # initialise plotting canvas

    cmap = plt.get_cmap("Accent")
    bounds = [-1, 0, 1, 2, 3]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    wr = np.ones(len(alphas))
    wr[-1] = 1
    fig, ax = plt.subplots(1, len(alphas), figsize=(3*len(alphas), 3), sharey=True, gridspec_kw={"width_ratios":wr})

    for i, a in enumerate(alphas):
        im = ax[i].imshow(number_of_desync[:,i].reshape((sides,sides)), extent=[x[0], x[-1], y[0], y[-1]], aspect="auto",
                       cmap=cmap, norm=norm, interpolation="nearest")
        ax[i].set_xlabel(r"$\phi_k$")
        if i == 0:
            ax[i].set_ylabel( r"$\omega_k$")
        ax[i].set_title(r"$\alpha={:.2g}$".format(a))

        #if i + 1 == len(alphas):
        #    ax[i + 1].set_visible(False)
        #    cbar = fig.colorbar(im, ax=ax[i+1], cmap=cmap, norm=norm, boundaries=bounds, label="", shrink=0.6)
        #    cbar.set_clim(-1, 2)
        #    cbar.set_ticks(list(range(-1, 3)))
        #    cbar.set_ticklabels(["" for t in range(4)])
        #    cbar.ax.text(1.2, .875, "other basins", ha='left', va='center', size=16)
        #    cbar.ax.text(1.2, .625, "solitary", ha='left', va='center', size=16)
        #    cbar.ax.text(1.2, .375, "phase sync", ha='left', va='center', size=16)
        #    cbar.ax.text(1.2, .125, "exotic solitary", ha='left', va='center', size=16)
        #    cbar.ax.yaxis.label.set_color("w")

    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, "pp_types.pdf".format(a)))
    plt.close(fig)

    fig, ax = plt.subplots(1, len(alphas), figsize=(3 * len(alphas), 3), sharey=True, gridspec_kw={"width_ratios": wr})

    for i, a in enumerate(alphas):
        ax[i].set_xlabel(r"$\phi_k$")
        if i == 0:
            ax[i].set_ylabel(r"$\omega_k$")

        ax[i].set_title(r"$\alpha={:.2g}$".format(a))

        im = ax[i].imshow(number_of_desync[:, i].reshape((sides,sides)), extent=[x[0], x[-1], y[0], y[-1]], aspect="auto",
                       cmap=cmap, interpolation="nearest", vmin=0, vmax=10)

        #if i + 1 == len(alphas):
        #    ax[i + 1].set_visible(False)
        #    fig.colorbar(im, ax=ax[i+1], extend="max", label=r"$n_{dedsync}$")

    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, "pp_noformat.pdf".format(a)))
    plt.close(fig)

def init_observables():
    observables = dict()
    observables["aspl"] = []
    observables["waspl"] = []
    observables["cc"] = []
    observables["wnd"] = []
    observables["nd"] = []
    observables["md"] = []
    observables["wmd"] = []
    observables["cfb"] = []
    observables["cfc"] = []
    observables["degree_count"] = []
    return observables

def update_observables(g, observables):
    number_of_nodes = g.number_of_nodes()
    observables["aspl"].append(nx.average_shortest_path_length(g, weight="weight"))
    observables["waspl"].append(nx.average_shortest_path_length(g))
    observables["cc"].append(nx.average_clustering(g))
    observables["wnd"].append(np.mean(list(nx.average_degree_connectivity(g, weight="weight").values())))
    observables["nd"].append(np.mean(list(nx.average_degree_connectivity(g).values())))
    observables["md"].append(np.mean(list(zip(*nx.degree(g)))[1]))
    observables["wmd"].append(np.mean(list(zip(*nx.degree(g, weight="weight")))[1]))
    observables["cfb"].append(
        sum(nx.current_flow_betweenness_centrality(g, weight="weight").values()) / number_of_nodes)
    observables["cfc"].append(
        sum(nx.current_flow_closeness_centrality(g, weight="weight").values()) / number_of_nodes)
    for d, count in enumerate(nx.degree_histogram(g)):
        for _ in range(count):
            observables["degree_count"].append(d)

# plots should only be created on the master process
@bao.run_on_master
def ensemble_statistics(dr, sim_dir):
    from .sim_functions import load_fps_from_file
    from .parameters_and_fixed_points import northern_admittance
    import glob

    bao.ensure_dir_exists(sim_dir)

    fps_folder = os.path.join(sim_dir, "fps/")
    nob = len(glob.glob1(fps_folder,"*.hdf"))

    canvas = ap.Plot(output="paper")
    canvas.figure_format = "png"

    observables = init_observables()
    for batch in range(nob):
        Y, _, _ = load_fps_from_file(fps_folder, batch, 0)
        g = nx.Graph(np.abs(Y))
        g.remove_edges_from(g.selfloop_edges())
        update_observables(g, observables)

    ## load northern grid data for comparison
    cp = dr["consumer/producer"]
    pk = dr["P/K"]
    dkh = dr["D^2/KH"]

    northern = init_observables()
    Y = northern_admittance(os.path.join("input_data", "northern.json"))
    Y /= 1. * pk * np.mean(np.abs(Y)[Y.nonzero()])
    g = nx.Graph(np.abs(Y))
    g.remove_edges_from(g.selfloop_edges())
    update_observables(g, northern)

    for i, obs in observables.items():
        if i is not "degree_count":
            fig = canvas.draw_hist({i: obs}, label=i, legend=False)
            fig.axes[0].axvline(x=northern[i][0], ymin=0, ymax=1, )
        else:
            fig = canvas.draw_hist({"ensemble": obs, "northern": northern[i]}, nbins=10, label=i, legend=True)
        canvas.save(os.path.join(fps_folder, i), fig)

# plots should only be created on the master process
@bao.run_on_master
def experiment_summary(dynamical_regimes, in_folder, suffix="_0"):
    data_folder = os.path.abspath(in_folder)
    grouped_asbs_plot(dynamical_regimes, folder=data_folder, suffix=suffix)
    grouped_pp_plot(dynamical_regimes,
                    folder=data_folder, suffix=suffix, alpha_idx=0)
    grouped_pp_plot(dynamical_regimes,
                    folder=data_folder, suffix=suffix, alpha_idx=1)
    grouped_pp_plot(dynamical_regimes,
                    folder=data_folder, suffix=suffix, alpha_idx=3)
    grouped_pp_plot(dynamical_regimes,
                    folder=data_folder, suffix=suffix, alpha_idx=-1)