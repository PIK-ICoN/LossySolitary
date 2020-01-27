#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Paul Schultz"
__date__ = "Feb 15, 2018"
__version__ = "v2.3"

"""
Module contains class AwesomePlot.

Derivative of the matplotlib module. The aim is to create
visually attractive, unambigous and colour-blind-friendly
images, customised for PIK corporate design.

The class has the no instance variables.

Overriden inherited methods::

    (NoneType)  show            : future changes to pyplot.show()

"""

#TODO: distribution-adjusted colormap
#TODO: magic function to modify style of existing figures
#TODO: label on bar in histogram
#TODO: map plots

# Import NumPy for the array object and fast numerics.
import numpy as np

# import matplotlib and submodules
import matplotlib
from matplotlib import pyplot
from matplotlib import cycler
from matplotlib.cm import register_cmap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, hex2color

# import seaborn for fancy style templates
import seaborn

# import additional colormaps https://github.com/BIDS/colormap
#from . import colormaps

# import warnings module to issue warnings on user input without interrupting the program
import warnings

# TODO: optional fig,ax objects as arguments: add_*(..., figax=None): fig, ax = figax

class Plot(object):
    """
    AwesomePlot class.

    The class is consistently sets reasonable matplotlib rc parameters for three different use cases.
    It can be instantiated either to create images for

    a) publications (A4, pdf)
    b) talks (A5, png)
    c) icons (not implemented)

    Images are landscape per default.
    """

    def __init__(self, output='paper', rc_spec={}, font_scale=2):
        """
            Initialise an instance of AwesomePlot.

            Parameters
            ----------
            output: string
                can be either "paper", "talk" or "icon"

            """

        assert output in ["paper", "talk", "poster", "notebook"]

        # workaround for KeyError in self.rc savefig.format
        if output in ["talk", "poster"]:
            self.figure_format = "png"
            self.transparent = True
        else:
            self.figure_format = "pdf"
            self.transparent = False

        self.rc = {'xtick.direction': 'in',
                   'ytick.direction': 'in',
                   'verbose.level': 'helpful',
                   'lines.linewidth': 3,
                   'axes.linewidth': 3
                   }

        if rc_spec:
            self.rc.update(rc_spec)

        seaborn.set_style(style="white", rc=self.rc)
        seaborn.set_context(output, font_scale=font_scale, rc=self.rc)

        # predefine colour maps:

        # generic discrete cmap (10 items)
        self.discrete_colours = ListedColormap(
            np.array(
                ['#1f77b4', '#33a02c', '#ff7f00', '#6a3d9a', '#e31a1c', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6']),
            'discrete'
        )
        register_cmap('discrete', cmap=self.discrete_colours)

        # PIK discrete cmap (4 items)
        self.pik_colours = ListedColormap(
            np.array(['#F25B28', '#009FDA', '#69923A', '#686C70']),
            'pik'
        )
        register_cmap(self.pik_colours.name, cmap=self.pik_colours)

        # linear interpolated cmap (based on PIK colours)
        self.lin_colours = LinearSegmentedColormap.from_list(
            'linear', [(0, 'w'), (1, hex2color('#e37222'))]
        )
        self.lin_colours.set_bad(hex2color('#8e908f'))
        register_cmap(self.lin_colours.name, cmap=self.lin_colours)

        # symmetric interpolated cmap (based on PIK colours)
        self.sym_colours = LinearSegmentedColormap.from_list(
            'sym', [(0, hex2color('#009fda')), (0.5, hex2color('#8e908f')), (1, hex2color('#e37222'))]
        )
        self.sym_colours.set_bad('k')
        register_cmap(self.sym_colours.name, cmap=self.sym_colours)

        # linestyle sequence for multiplots
        self.linestyles = np.tile(['-', '--', '-.', ':'], 1 + self.discrete_colours.N // 4)[:self.discrete_colours.N]

        self.set_default_colours('pik')

        self.figures = []

        # function aliases for backwards compatibility
        self.add_contour = self.draw_contour
        self.add_distplot = self.draw_distplot
        self.add_hist = self.draw_hist
        self.add_lineplot = self.draw_lineplot
        self.add_network = self.draw_network
        self.add_scatterplot = self.draw_scatterplot

    @classmethod
    def paper(cls, font_scale=1.2):
        """
        Class method yielding an AwesomePlot instance of type "paper"

        Parameters
        ----------
        cls: object
            AwesomePlot class

        Returns
        -------
        instance of class AwesomePlot

        """

        rc = dict()
        rc['figure.figsize'] = (11.69, 8.268)  # A4
        rc['pdf.compression'] = 6  # 0 to 9
        rc['savefig.format'] = 'pdf'
        rc['pdf.fonttype'] = 42
        rc['savefig.dpi'] = 300

        return cls(output='paper', rc_spec=rc, font_scale=font_scale)

    @classmethod
    def talk(cls, font_scale=1.2):
        """
        Class method yielding an AwesomePlot instance of type "talk"

        Parameters
        ----------
        cls: object
            AwesomePlot class

        Returns
        -------
        instance of class AwesomePlot

        """
        rc = dict()
        rc['figure.figsize'] = (8.268, 5.872)  # A5
        rc['savefig.format'] = 'png'
        rc['savefig.dpi'] = 300

        return cls(output='talk', rc_spec=rc, font_scale=font_scale)

    @classmethod
    def poster(cls, font_scale=1.2):
        """
        Class method yielding an AwesomePlot instance of type "poster"

        Parameters
        ----------
        cls: object
            AwesomePlot class

        Returns
        -------
        instance of class AwesomePlot

        """

        rc = dict()
        rc['savefig.format'] = 'png'
        rc['savefig.dpi'] = 300

        return cls(output='poster', rc_spec=rc, font_scale=font_scale)

    @classmethod
    def notebook(cls, font_scale=1.2):
        """
        Class method yielding an AwesomePlot instance of type "notebook"

        Parameters
        ----------
        cls: object
            AwesomePlot class

        Returns
        -------
        instance of class AwesomePlot

        """

        rc = dict()
        rc['savefig.format'] = 'png'
        rc['savefig.dpi'] = 300

        return cls(output='notebook', rc_spec=rc, font_scale=font_scale)

    ###############################################################################
    # ##                       PUBLIC FUNCTIONS                                ## #
    ###############################################################################


    def draw_lineplot(self, figax=None, x=None, lines={}, shades={}, labels=['x', 'y'], marker="o", linewidth=None,
                     sortfunc=None, grid=False, infer_layout=True, legend=True):
        """
        Plots (multiple) lines with optional shading.

        This function adds a matplotlib figure object to the figure collection of
        an AwesomePlot instance.

        Parameters
        ----------
        x: numpy ndarray (optional)
            array with x-values common to all lines
        lines:  dict
            dictionary of type {key: y} containing y-values,
            multiple lines need to be distinguished by unique keys
        shades: dict (optional)
            dictionary of type {key: [y - l, y +  u]} containing upper and lower
            intervals to indicate uncertainty, confidence intervals etc.
        labels: list [str]
            list containing  meaningful axis labels
        sortfunc: function or lambda expression
            optionally supply a function that is used to sort the line keys for plotting
            e.g. (a) sortfunc = float (b) sortfunc=f, where f = lambda x: float(x.split()[-2])
        grid: bool
            if true, background grid is drawn
        infer_layout: bool
            if false min and max will not be set , important for plots with NANs and Infs
        """

        assert len(labels) == 2
        # the next line leads to an error if there are more lines to be plotted
        # than colours available, although there are also different linestyles possible
        # TODO write new assert with message that you can use other colourmaps by p.set_default_colours(x),
        # with x = 'discrete'(10 colours), 'pik' (4 colours, default), 'linear', 'sym'
        #assert len(lines.keys()) <= self.dfcmp.N

        if x is None:
            x = np.arange(len(lines[list(lines.keys())[0]]))

        if linewidth is None:
            linewidth = self.rc['lines.linewidth']

        # if shades:
        #     assert set(shades.keys()).issubset(lines.keys())

        if figax is None:
            fig, ax = pyplot.subplots(nrows=1, ncols=1)
        else:
            fig, ax = figax

        # determine boundaries and scale
        if infer_layout:
            xmin = np.min(list(x.values())) if isinstance(x, dict) else np.min(x)
            xmax = np.max(list(x.values())) if isinstance(x, dict) else np.max(x)
            if not shades:
                ymin = np.min([np.min(l) for l in list(lines.values())])
                ymax = np.max([np.max(l) for l in list(lines.values())])
            else:
                ymin = np.min([np.min(l) for l in list(shades.values())])
                ymax = np.max([np.max(l) for l in list(shades.values())])

            xmargin = (xmax - xmin) / 200.
            ymargin = (ymax - ymin) / 200.

            scale = np.log(1 + 1. * (xmax - xmin) / len(x))

            ax.axis([xmin - xmargin, xmax + xmargin, ymin - ymargin, ymax + ymargin])

        if grid:
            ax.grid()

        for i in sorted(list(lines.keys()), key=sortfunc, reverse=False):
            _x = x[i] if isinstance(x, dict) else x
            if shades and i in list(shades.keys()):
                shade = ax.fill_between(_x, shades[i][0], shades[i][1], alpha=0.3, edgecolor='none', facecolor=hex2color('#8E908F'))
                if infer_layout:
                    ax.plot(_x, lines[i], linewidth=linewidth, marker=marker, mew=3. * scale, mec=shade._facecolors[0], ms=10. * scale, label=i)
                else:
                    ax.plot(_x, lines[i], linewidth=linewidth, marker=marker, mec=shade._facecolors[0], label=i)
            else:
                if infer_layout:
                    ax.plot(_x, lines[i], linewidth=linewidth, marker=marker, mec='w', mew=3 * scale, ms=10. * scale, label=i)
                else:
                    ax.plot(_x, lines[i], linewidth=linewidth, marker=marker, mec='w', label=i)

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        if legend:
            pyplot.legend(frameon=True)
        #fig.tight_layout()

        self.figures.append(fig)

        return fig

    def draw_distplot(self, x, y, labels=['x', 'y'], linestyle='-', filled=True, text=True, marker="o"):

        assert len(labels) == 2

        from scipy.stats.mstats import mquantiles

        m = np.mean(y, axis=1)
        q = mquantiles(y, axis=1, prob=list([.05, .25, .75, .95]))

        # determine boundaries
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = y.min()
        ymax = y.max()

        xmargin = (xmax - xmin) / 200.
        ymargin = (ymax - ymin) / 200.

        scale = np.log(1 + .5 * (xmax - xmin) / len(x))

        fig, ax = pyplot.subplots(nrows=1, ncols=1)

        ax.axis([xmin - xmargin, xmax + xmargin, ymin - ymargin, ymax + ymargin])

        if filled:
            ax.fill_between(x, q[:, 0],
                            q[:, 3], facecolor=hex2color('#8E908F'), edgecolor='none', alpha=0.3)
            ax.fill_between(x, q[:, 1],
                            q[:, 2], facecolor=hex2color('#8E908F'), edgecolor='none', alpha=0.3)
        else:
            ax.plot(x, q[:, 1], color='r', linestyle='--', label='25\%')
            ax.plot(x, q[:, 2], color='r', linestyle='--', label='75\%')
            ax.plot(x, q[:, 0], color='r', linestyle=':', label='5\%')
            ax.plot(x, q[:, 3], color='r', linestyle=':', label='95\%')

        if text:
            ax.text(x[1], q[1, 0] * .95, r'5%') #, fontsize=.6 * self.textsize)
            ax.text(x[2], q[2, 1] * .95, r'25%') #, fontsize=.6 * self.textsize)
            ax.text(x[3], q[3, 2] * .95, r'75%') #, fontsize=.6 * self.textsize)
            ax.text(x[4], q[4, 3] * .95, r'95%') #, fontsize=.6 * self.textsize)

        ax.plot(x, m, marker=marker, mec='w', mew=3*scale, ms=10*scale)

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        fig.tight_layout()

        self.figures.append(fig)

        return fig

    def draw_contour(self, x, y, z, labels=['x', 'y', 'z'], nlevel=10, cmap=None,
                    text=False, horizontal=False, pi=None, layout=True, fixed_scale=None, boundary=True,
                    colorbar=False):
        """
            Plots Contourplots

            Parameters
            ----------
            x: array
                x-values associated to the entries in z
            y: array
                y-values associated to the entries in z
            z: matrix
                data of shape [len(x), len(y)] containing values for all (x, y) pairs
            nlevel: int
                number of levels of the contourplot
            sym: bool
                False: using linear colour scale, else diverging
            text: bool
                True: labels on every level line
            horizontal: bool
                True: horizontal colour bar
            pi: "xaxis" or "yaxis"
                if one of the axis is given in multiples of pi
            layout: bool
                False means, the contourf/contour function dont get a
                number of levels or a zmin and zmax. This is necessary
                for a matrix z with NANs or Infs in it, since then
                zmin and zmax become NAN/Inf.
            fixed_scale: tuple
                min/max values to apply a fixed colour scale to z-values
            boundary: bool
                whether to draw contour lines or not
            colorbar: bool
                draw a colorbar or not

        """
        assert len(labels) == 3

        # Issue warning if z contains NaN or Inf
        if not np.isfinite(z).all():
            warnings.warn("Since z is not finite, it would be better to use layout=False.")

        if cmap is None:
            colormap = pyplot.get_cmap('linear')
        else:
            colormap = pyplot.get_cmap(cmap)

        backup = matplotlib.rcParams['lines.linewidth']
        matplotlib.rcParams['lines.linewidth'] = 1

        # determine boundaries
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()

        fig, ax = pyplot.subplots(nrows=1, ncols=1)

        fig.tight_layout()

        if not fixed_scale is None:
            zmin, zmax = fixed_scale
        else:
            zmin = np.floor(z[np.isfinite(z)].min())
            zmax = z[np.isfinite(z)].max()

            if zmax > 0.5:
                zmax = np.ceil(z[np.isfinite(z)].max())

            if zmin == zmax:
                zmax += 0.5

        pyplot.gca().patch.set_color('#8e908f')  # print the Nan/inf Values in black)

        if layout:
            levels = np.linspace(zmin, zmax, nlevel + 1, endpoint=True)
            c = ax.contourf(x, y, z, levels=levels, cmap=cmap, origin='lower', antialiased=True, vmin=zmin, vmax=zmax)
            if boundary:
                cl = ax.contour(x, y, z, colors='k', levels=levels)
        else:
            c = ax.contourf(x, y, z, cmap=cmap, origin='lower', antialiased=True, vmin=zmin, vmax=zmax)
            if boundary:
                cl = ax.contour(x, y, z, colors='k')

        if text:
            ax.clabel(cl, fontsize=.25 * self.textsize, inline=1)

        ax.axis([xmin, xmax, ymin, ymax])

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        if pi is not None:
            if pi == "xaxis":
                ticklabels = np.empty(np.size(ax.get_xticks()), dtype='object')
                ticks = ax.get_xticks()
            elif pi == "yaxis":
                ticklabels = np.empty(np.size(ax.get_yticks()), dtype='object')
                ticks = ax.get_yticks()
            else:
                raise ValueError()

            for i, tick in enumerate(ticks):
                if np.isclose(tick, -np.pi, atol=0.05):
                    ticklabels[i] = "$-\pi$"
                elif np.isclose(tick, np.pi, atol=0.05):
                    ticklabels[i] = "$\pi$"
                elif tick == 0:
                    ticklabels[i] = str(0)
                else:
                    ticklabels[i] = "{0:.1g}".format(tick / np.pi) + "$\pi$"

            if pi == "xaxis":
                ax.set_xticklabels(ticklabels)
            elif pi == "yaxis":
                ax.set_yticklabels(ticklabels)

        if colorbar and horizontal:
            fig.colorbar(c, label=labels[2], orientation='horizontal', pad=0.2)
        elif colorbar:
            fig.colorbar(c, label=labels[2])  # not so cool for smalll numbers format=r"%.1f"

        self.figures.append(fig)

        matplotlib.rcParams['lines.linewidth'] = backup

        return fig


    def draw_scatterplot(self, x, y, labels=['x', 'y'], factor=None, show_annot=None, bins=20, kind="scatter", kdeplot=False, c_map="linear"):
        assert len(labels) == 2

        if isinstance(x, dict):
            assert sorted(x.keys()) == sorted(y.keys())
            assert len(list(x.keys())) <= self.dfcmp.N
            # determine boundaries
            xmin = np.min([np.min(l) for l in list(x.values())])
            xmax = np.max([np.max(l) for l in list(x.values())])
            ymin = np.min([np.min(l) for l in list(y.values())])
            ymax = np.max([np.max(l) for l in list(y.values())])
        else:
            # determine boundaries
            xmin = x.min()
            xmax = x.max()
            ymin = y.min()
            ymax = y.max()

        # adjust colors
        if kdeplot:
            c = "k"
        else:
            if factor is None:
                c = self.pik_colours.colors[0]
            else:
                c = factor

        settings = {
            "joint_kws": dict(alpha=1, color=c, cmap=pyplot.get_cmap(c_map)),
            "marginal_kws": dict(bins=bins, rug=False),
            "annot_kws": dict(stat=None, frameon=True, loc="best", handlelength=0),
            "space": 0.1,
            "kind": kind,
            "xlim": (xmin, xmax),
            "ylim": (ymin, ymax)
        }

        scatter = seaborn.jointplot(x, y, stat_func=show_annot, **settings)

        if kdeplot:
            scatter.plot_joint(seaborn.kdeplot, shade=True, cut=5, zorder=0, n_levels=6, cmap=pyplot.get_cmap(c_map))

        scatter.set_axis_labels(*labels)

        self.figures.append(scatter.fig)

        return scatter.fig

    def draw_hist(self, data, label='x', nbins=20, sortfunc=None, legend=True):

        # ensure data is dict
        assert isinstance(data, dict)

        xmin = np.min([np.min(l) for l in list(data.values())])
        xmax = np.max([np.max(l) for l in list(data.values())])

        xmargin = (xmax - xmin) / 100.

        fig, ax = pyplot.subplots(nrows=1, ncols=1)

        fig.tight_layout()

        bottom = np.zeros(nbins)
        ymax = 0.
        counter = 0
        _, b = np.histogram(data[list(data.keys())[0]], bins=nbins, density=True)
        for d in sorted(list(data.keys()), key=sortfunc, reverse=False):
            c = list(matplotlib.rcParams['axes.prop_cycle'])[counter]['color']
            h, _ = np.histogram(data[d], bins=nbins, density=True)
            ax.bar(.5 * (b[1:] + b[:-1]), h, bottom=bottom, color=c, edgecolor="none", align='center', zorder=1,
                   width=(xmax - xmin) / (nbins * 1.5), label=d)
            bottom += h
            counter += 1
            ymax += h.max()

        ax.set_xlim([xmin - xmargin, xmax + xmargin])
        ax.set_ylim([0., ymax * 1.1])
        ax.set_xlabel(label)
        ax.set_ylabel(r'density')

        ax.yaxis.grid(color='w', linestyle='-', zorder=2)

        if legend:
            pyplot.legend(frameon=True)
        fig.tight_layout()

        self.figures.append(fig)

        return fig

    def draw_network(self, adjacency, styles={}, axis_labels=None, vertex_labels=None, labels=False, height=False, node_cb=False, edge_cb=False, node_cmap=None, edge_cmap=None):
        """
            Plots network, submit eg vertex color values via styles={"vertex_color":values}

            Parameters
            ----------
            adjacency:
            styles: dict

            sym: bool
                if true:
            axis_labels:

            vertex_labels: array
                e.g. pars["input_power"]
            labels: bool

            height: bool


        """
        if height:
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Line3DCollection as LineCollection
        else:
            from matplotlib.collections import LineCollection
        from scipy.sparse import issparse, isspmatrix_dok

        if issparse(adjacency):
            assert isspmatrix_dok(adjacency)
            # print "Build network from sparse dok matrix."
            N = adjacency.shape[0]
            edgelist = sorted(set([tuple(np.sort(key)) for key in list(adjacency.keys())]))
        else:
            N = len(adjacency)
            edgelist = np.vstack(np.where(adjacency > 0)).transpose()
            edgelist = sorted(set([tuple(np.sort(edgelist[i])) for i in range(len(edgelist))]))

        source = [e[0] for e in edgelist]
        target = [e[1] for e in edgelist]

        if node_cmap is None:
            node_cmap = pyplot.get_cmap("linear")

        if edge_cmap is None:
            edge_cmap = pyplot.get_cmap("linear")


        visual_style = dict(
            edge_color=np.repeat('#8e908f', len(edgelist)),
            edge_width=seaborn.axes_style()["axes.linewidth"],
            vertex_size=100,
            vertex_label=list(range(N))
        )

        if styles:
            visual_style.update(styles)

        if "layout" not in visual_style:
            if height:
                visual_style["layout"] = np.random.random([N, 3])
            else:
                visual_style["layout"] = np.random.random([N, 2])
            print("Assign random layout for plotting.")


        if "edge_color_dict" in visual_style:
            min_color = np.min(list(visual_style["edge_color_dict"].values()))
            max_color = np.max(list(visual_style["edge_color_dict"].values()))
            f = lambda x: (np.float(visual_style["edge_color"][x]) - min_color) / (max_color - min_color)
            visual_style["edge_color"] = [f(e) for e in edgelist]
            alpha = 1.
        else:
            alpha = 1.


        if height:
            fig = pyplot.figure()
            ax = fig.gca(projection='3d')
            x, y, z = list(zip(*visual_style["layout"]))
            args = (x, y, z)
        else:
            fig, ax = pyplot.subplots(nrows=1, ncols=1)
            fig.tight_layout()
            x, y = list(zip(*visual_style["layout"]))
            args = (x, y)

        # ax.axis("off")

        if height:
            xyz = (np.asarray(((visual_style["layout"][source, 0],
                               visual_style["layout"][source, 1],
                               visual_style["layout"][source, 2]),
                              (visual_style["layout"][target, 0],
                               visual_style["layout"][target, 1],
                               visual_style["layout"][target, 2]))
                             ).transpose(2, 0, 1))
        else:
            xyz = (np.asarray(((visual_style["layout"][source, 0],
                                visual_style["layout"][source, 1]),
                               (visual_style["layout"][target, 0],
                                visual_style["layout"][target, 1]))
                              ).transpose(2, 0, 1))
        l_collection = LineCollection(xyz,
                                      linewidths=visual_style["edge_width"],
                                      antialiaseds=(1,),
                                      colors=visual_style["edge_color"],
                                      cmap=edge_cmap,
                                      alpha=alpha,
                                      zorder=1,
                                      transOffset=ax.transData)
        ax.add_collection(l_collection)

        # if edge_cb:
        #TODO: edge colorbar
        #if visual_style.has_key("edge_color_dict"):
        #    sm = pyplot.cm.ScalarMappable(cmap=map_edges, norm=pyplot.Normalize(vmin= min_color, vmax= max_color))
        #    # fake up the array of the scalar mappable. Urgh...
        #    sm.set_array(visual_style["edge_color"])
        #    cb= pyplot.colorbar(sm,format=r"%.2f")
        #    cb.outline.set_visible(False)
        #    from matplotlib import ticker
        #    tick_locator = ticker.MaxNLocator(nbins=6)
        #    cb.locator = tick_locator
        #    cb.update_ticks()
        #    ax.set_title('maximum equals '+str(max_color)+' at edge '+str(visual_style["edge_color_dict"].keys()[np.argmax(visual_style[
        # "edge_color_dict"].values())]))


        margin = max(0.05 * (np.max(x) - np.min(x)), 0.05 * (np.max(y) - np.min(y)))
        ax.set_xlim([np.min(x) - margin, np.max(x) + margin])
        ax.set_ylim([np.min(y) - margin, np.max(y) + margin])

        if "vertex_color" not in visual_style:
            nodes = ax.scatter(*args,
                               c='#8e908f',
                               s=visual_style["vertex_size"],
                               cmap=node_cmap,
                               edgecolor='w',
                               zorder=2)
        else:
            nodes = ax.scatter(*args,
                               c=visual_style["vertex_color"],
                               s=visual_style["vertex_size"],
                               cmap=node_cmap,
                               vmin=np.floor(np.min(visual_style["vertex_color"])),
                               vmax=np.ceil(np.max(visual_style["vertex_color"])),
                               edgecolor='w',
                               zorder=2)
            if node_cb:
                cb = fig.colorbar(nodes, orientation='horizontal', shrink=0.66, format=r"%.2f")

        if axis_labels:
            ax.set_xlabel(axis_labels[0], labelpad=30)
            ax.set_ylabel(axis_labels[1], labelpad=30)
            if height:
                ax.set_zlabel(axis_labels[2], labelpad=30)

        if vertex_labels is None:
            if labels:
                for i in range(N):
                    pyplot.annotate(str(i), xy=(x[i], y[i]), xytext=(3, 3), textcoords='offset points',
                                    # size=0.5 * self.rc["font.size"],
                                    horizontalalignment='left', verticalalignment='bottom')
        else:
            for i in range(N):
                pyplot.annotate(str(vertex_labels[i]), xy=(x[i], y[i]), xytext=(3, -25),
                                textcoords='offset points',
                                # size=0.5 * self.params["font.size"],
                                horizontalalignment='left', verticalalignment='bottom')

        # we may adjust the background colour to make light nodes more visible
        #ax.set_axis_bgcolor((.9, .9, .9))

        self.figures.append(fig)

        return fig

    def space_time_plot(self, x, y, z, labels=['x', 'y', 'z'], zmin=0, zmax=1, colorbar=True, sym=False, horizontal=False, pi=None):

        warnings.warn("New and not much tested.")

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

        warnings.warn("New and not much tested.")

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

        warnings.warn("New and not much tested.")

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

    def save(self, fnames, fig = None):
        if fig:
            fig.savefig(fnames + '.' + self.figure_format, bbox_inches='tight', transparent=self.transparent)
            self.clear(fig)
        else:
            assert len(fnames) == len(self.figures)
            for i, fig in enumerate(self.figures):
                print(("save:", fnames[i] + '.' + self.figure_format))
                fig.savefig(fnames[i] + '.' + self.figure_format, bbox_inches='tight', transparent=self.transparent)
                pyplot.close(fig)
            for i, fig in enumerate(self.figures):
                self.figures.remove(fig)


    def clear(self, fig):
        assert isinstance(fig, pyplot.Figure)
        pyplot.close(fig)
        self.figures.remove(fig)


    def show(self):
        if len(self.figures) > 0:
            pyplot.show()
        else:
            pass
            #raise ValueError("No open figures to plot.")


    def show_cmap(self, cmap_name):
        if isinstance(cmap_name, str):
            cmap = pyplot.get_cmap(cmap_name)
        else:
            cmap = cmap_name
        a=np.outer(np.arange(0,1,0.01), np.ones(10))
        pyplot.figure(figsize=(2,10))
        pyplot.axis("off")
        pyplot.imshow(a,aspect='auto',interpolation='nearest', cmap=cmap)
        pyplot.title(cmap_name,fontsize=15)
        pyplot.show()

    def update_params(self, dic):
        assert all([key in list(matplotlib.rcParams.keys()) for key in list(dic.keys())])
        self.rc.update(dic)
        # do not use seaborn.set(), this overrides all rc params ...
        for key, val in list(dic.items()):
            matplotlib.rcParams[key] = val


    def portrait(self):
        canvas = self.rc['figure.figsize']
        if canvas[1] > canvas[0]:
            warnings.warn("Figure is already in portrait orientation.")
        else:
            self.rc['figure.figsize'] = canvas[::-1]

    def set_default_colours(self, cmap_name):
        self.dfcmp = pyplot.get_cmap(cmap_name)

        # select m evenly spaced colours out of a list of n (https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm)
        idx = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]

        # cyclers need to be of equal length
        if len(self.linestyles) > self.dfcmp.N:
            self.update_params(
                {
                    'axes.prop_cycle': cycler('color', list(self.dfcmp.colors)) + \
                                       cycler('linestyle', self.linestyles[:self.dfcmp.N]),
                    'image.cmap': self.dfcmp.name
                }
            )
        else:
            selection = idx(len(self.linestyles), self.dfcmp.N)
            self.update_params(
                {
                    'axes.prop_cycle': cycler('color', np.array(self.dfcmp.colors)[selection]) + \
                                       cycler('linestyle', self.linestyles),
                    'image.cmap': self.dfcmp.name
                }

            )
        #TODO color cycler with selected colours

    def set_log(self, fig, log="y"):
        assert fig in self.figures
        if log == "y":
            fig.axes[0].set_yscale('symlog')
        elif log == "x":
            fig.axes[0].set_xscale('symlog')
        elif log == "xy":
            fig.axes[0].set_xscale('symlog')
            fig.axes[0].set_yscale('symlog')
        else:
            raise ValueError("Invalid input. Must be x, y, or xy.")


class AddonPandas(object):
    """
    Plot class.

    The class is consistently sets reasonable matplotlib rc parameters for three different use cases.
    It can be instantiated either to create images for

    a) publications (A4, pdf)
    b) talks (A5, png)
    c) posters (not implemented)

    Images are landscape per default.
    """

    # def __init__(self, output='paper', rc_spec={}, font_scale=1.1, use_pandas=False):
    #     super(AddonPandas, self).__init__(output, rc_spec, font_scale)
    #     self.use_pandas = use_pandas


    ###############################################################################
    # ##                       PUBLIC FUNCTIONS                                ## #
    ###############################################################################

    @staticmethod
    def df_to_dict(df):
        from pandas import DataFrame
        assert isinstance(df, DataFrame)

        # create dict from dataframe
        dd = df.to_dict()

        # remove sub-dictionaries
        return {key: list(dd[key].values()) for key in list(dd.keys())}

    ###############################################################################
    # ##                      PRIVATE FUNCTIONS                                ## #
    ###############################################################################


    def draw_lineplot(self, df, firstcol=False, legend=True, grid=True, logx=False, logy=False, loglog=False):
        # transfer x-values to dataframe index
        if firstcol:
            df.index = df[df.columns[0]]
            df = df[df.columns[1:]]


        fig, ax = pyplot.subplots(nrows=1, ncols=1)
        df.plot(
            kind="line",
            use_index=True,
            marker='o',
            mew=2,
            mec='w',
            colormap=self.dfcmp,
            grid=grid,
            legend=legend,
            ax=ax,
            loglog=loglog,
            logx=logx,
            logy=logy
        )

        self.figures.append(fig)

        return fig

    def draw_scatterplot(self, df, x, y, factor=None, bins=20, show_annot=None, kind="scatter", kdeplot=False, c_map="linear"):

        # FIXME: check, whether x and y are columns of df
        assert isinstance(x, str)
        assert isinstance(y, str)

        if kdeplot:
            c = "k"
        else:
            if factor is None:
                c = self.pik_colours.colors[0]
            else:
                assert isinstance(factor, str)
                c = df[factor]

        xmin = df[x].min()
        xmax = df[x].max()
        ymin = df[y].min()
        ymax = df[y].max()

        settings = {
            "joint_kws": dict(alpha=1, c=c, cmap=pyplot.get_cmap(c_map)),
            "marginal_kws": dict(bins=bins, rug=False),
            "annot_kws": dict(stat=r"r", frameon=True, loc=0, handlelength=0),
            "space": 0.1,
            "kind": kind,
            "xlim": (xmin, xmax),
            "ylim": (ymin, ymax)
        }

        try:
            scatter = seaborn.jointplot(x, y, data=df, stat_func=show_annot, **settings)
        except:
            # some kws are not valid in certain plot kinds
            pyplot.close()
            settings = {
                "annot_kws": dict(stat=r"r", frameon=True, loc=0, handlelength=0),
                "space": 0.1,
                "kind": kind,
                "xlim": (xmin, xmax),
                "ylim": (ymin, ymax)
            }
            scatter = seaborn.jointplot(x, y, data=df, stat_func=show_annot, **settings)

        if kdeplot:
            scatter.plot_joint(seaborn.kdeplot, shade=True, cut=5, zorder=0, n_levels=6, cmap=pyplot.get_cmap(c_map))

        fig = scatter.fig
        pyplot.close() # close JointGrid object

        self.figures.append(fig)

        return fig

    def draw_hist(self, df, columns=None, normed=True, nbins=20, log=False, c_map="pik"):

        if columns:
            df = df[columns]

        settings = {
            "stacked": True,
            #"alpha": 0.75,
            "bins": nbins,
            "normed": normed,
            "log": log,
            "cmap": c_map
        }

        fig, ax = pyplot.subplots(nrows=1, ncols=1)
        df.plot.hist(ax=ax, **settings)

        self.figures.append(fig)

        return fig

class PandasPlot(AddonPandas,Plot):
    pass


if __name__ == "__main__":
    # TODO: implement tests!
    import doctest
    doctest.testmod()

