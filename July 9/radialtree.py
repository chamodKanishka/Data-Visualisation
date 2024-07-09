import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import matplotlib.cm as cm
from matplotlib.axes import Axes
import matplotlib
from matplotlib.lines import Line2D

colormap_list = [
    "nipy_spectral",
    "terrain",
    "gist_rainbow",
    "CMRmap",
    "coolwarm",
    "gnuplot",
    "gist_stern",
    "brg",
    "rainbow",
]

def radialTree(
    Z2, fontsize=8, ax: Axes = None, palette="gist_rainbow",
    addlabels=True, sample_classes=None, colorlabels=None, colorlabels_legend=None,
):
    if ax is None:
        ax: Axes = plt.gca()
    linewidth = 5
    alpha = 1.0
    R = 1
    width = R * 0.1
    space = R * 0.05

    if colorlabels:
        offset = width * len(colorlabels) / R + space * (len(colorlabels) - 1) / R + 0.05
    elif sample_classes:
        offset = width * len(sample_classes) / R + space * (len(sample_classes) - 1) / R + 0.05
    else:
        offset = 0

    xmax = np.amax(Z2["icoord"])
    xmin = np.amin(Z2["icoord"])
    ymax = np.amax(Z2["dcoord"])
    ucolors = sorted(set(Z2["color_list"]))
    cmp = cm.get_cmap(palette, len(ucolors))

    if type(cmp) == matplotlib.colors.LinearSegmentedColormap:
        cmap = cmp(np.linspace(0, 1, len(ucolors)))
    else:
        cmap = cmp.colors

    for icoord, dcoord, c in sorted(zip(Z2["icoord"], Z2["dcoord"], Z2["color_list"])):
        _color = cmap[ucolors.index(c)]
        if c == "C0":
            _color = "black"

        r = R * (1 - np.array(dcoord) / ymax)
        _x = np.cos(2 * np.pi * np.array([icoord[0], icoord[2]]) / xmax)
        _xr0, _xr1 = _x[0] * r[0], _x[0] * r[1]
        _xr2, _xr3 = _x[1] * r[2], _x[1] * r[3]
        _y = np.sin(2 * np.pi * np.array([icoord[0], icoord[2]]) / xmax)
        _yr0, _yr1 = _y[0] * r[0], _y[0] * r[1]
        _yr2, _yr3 = _y[1] * r[2], _y[1] * r[3]

        ax.plot([_xr0, _xr1], [_yr0, _yr1], c=_color, linewidth=linewidth, alpha=alpha)
        ax.plot([_xr2, _xr3], [_yr2, _yr3], c=_color, linewidth=linewidth, alpha=alpha)

        if _yr1 > 0 and _yr2 > 0:
            link = np.sqrt(r[1] ** 2 - np.linspace(_xr1, _xr2, 100) ** 2)
            ax.plot(np.linspace(_xr1, _xr2, 100), link, c=_color, linewidth=linewidth, alpha=alpha)
        elif _yr1 < 0 and _yr2 < 0:
            link = -np.sqrt(r[1] ** 2 - np.linspace(_xr1, _xr2, 100) ** 2)
            ax.plot(np.linspace(_xr1, _xr2, 100), link, c=_color, linewidth=linewidth, alpha=alpha)
        elif _yr1 > 0 and _yr2 < 0:
            _r = r[1]
            if _xr1 < 0 or _xr2 < 0:
                _r = -_r
            link1 = np.sqrt(_r ** 2 - np.linspace(_xr1, _r, 100) ** 2)
            ax.plot(np.linspace(_xr1, _r, 100), link1, c=_color, linewidth=linewidth, alpha=alpha)
            link2 = -np.sqrt(_r ** 2 - np.linspace(_r, _xr2, 100) ** 2)
            ax.plot(np.linspace(_r, _xr2, 100), link2, c=_color, linewidth=linewidth, alpha=alpha)

    label_coords = []
    for i, label in enumerate(Z2["ivl"]):
        place = (5.0 + i * 10.0) / xmax * 2
        label_coords.append([
            np.cos(place * np.pi) * (1.05 + offset),
            np.sin(place * np.pi) * (1.05 + offset),
            place * 180,
        ])

    if addlabels:
        assert len(Z2["ivl"]) == len(label_coords), (
            f'Internal error, label numbers for Z2 ({len(Z2["ivl"])})'
            f" and for calculated labels ({len(label_coords)}) must be equal!"
        )
        for (_x, _y, _rot), label in zip(label_coords, Z2["ivl"]):
            ax.text(
                _x, _y, label,
                {"va": "center"},
                rotation_mode="anchor",
                rotation=_rot,
                fontsize=fontsize,
            )

    if colorlabels:
        assert len(Z2["ivl"]) == len(label_coords), (
            "Internal error, label numbers "
            + str(len(Z2["ivl"]))
            + " and "
            + str(len(label_coords))
            + " must be equal!"
        )

        j = 0
        outerrad = R * 1.05 + width * len(colorlabels) + space * (len(colorlabels) - 1)

        intervals = []
        for i in range(len(label_coords)):
            _xl, _yl, _rotl = label_coords[i - 1]
            _x, _y, _rot = label_coords[i]
            if i == len(label_coords) - 1:
                _xr, _yr, _rotr = label_coords[0]
            else:
                _xr, _yr, _rotr = label_coords[i + 1]
            d = ((_xr - _xl) ** 2 + (_yr - _yl) ** 2) ** 0.5
            intervals.append(d)
        colorpos = intervals

        labelnames = []
        for labelname, colorlist in colorlabels.items():
            colorlist = np.array(colorlist)[Z2["leaves"]]
            outerrad = outerrad - width * j - space * j
            innerrad = outerrad - width
            patches, texts = ax.pie(
                colorpos,
                colors=colorlist,
                radius=outerrad,
                counterclock=True,
                startangle=label_coords[0][2] * 0.5,
                wedgeprops=dict(
                    width=width,
                ),
            )

            labelnames.append(labelname)
            j += 1

        if colorlabels_legend:
            for i, labelname in enumerate(labelnames):
                colorlines = []
                for c in colorlabels_legend[labelname]["colors"]:
                    colorlines.append(Line2D([0], [0], color=c, lw=4))
                leg = ax.legend(
                    colorlines,
                    colorlabels_legend[labelname]["labels"],
                    bbox_to_anchor=(1.5 + 0.3 * i, 1.0),
                    title=labelname,
                )
                ax.add_artist(leg)

    elif sample_classes:
        assert len(Z2["ivl"]) == len(label_coords), (
            "Internal error, label numbers "
            + str(len(Z2["ivl"]))
            + " and "
            + str(len(label_coords))
            + " must be equal!"
        )

        j = 0
        outerrad = R * 1.05 + width * len(sample_classes) + space * (len(sample_classes) - 1)

        intervals = []
        for i in range(len(label_coords)):
            _xl, _yl, _rotl = label_coords[i - 1]
            _x, _y, _rot = label_coords[i]
            if i == len(label_coords) - 1:
                _xr, _yr, _rotr = label_coords[0]
            else:
                _xr, _yr, _rotr = label_coords[i + 1]
            d = ((_xr - _xl) ** 2 + (_yr - _yl) ** 2) ** 0.5
            intervals.append(d)
        colorpos = intervals

        labelnames = []
        colorlabels_legend = {}
        for labelname, colorlist in sample_classes.items():
            ucolors = sorted(list(np.unique(colorlist)))
            type_num = len(ucolors)
            _cmp = cm.get_cmap(colormap_list[j], type_num)
            _colorlist = [_cmp(ucolors.index(c)) for c in colorlist]
            _colorlist = np.array(_colorlist)[Z2["leaves"]]
            outerrad = outerrad - width * j - space * j
            innerrad = outerrad - width
            patches, texts = ax.pie(
                colorpos,
                colors=_colorlist,
                radius=outerrad,
                counterclock=True,
                startangle=label_coords[0][2] * 0.5,
                wedgeprops=dict(
                    width=width,
                ),
            )

            labelnames.append(labelname)
            colorlabels_legend[labelname] = {}
            colorlabels_legend[labelname]["colors"] = _cmp(np.linspace(0, 1, type_num))
            colorlabels_legend[labelname]["labels"] = ucolors
            j += 1

        if colorlabels_legend:
            for i, labelname in enumerate(labelnames):
                colorlines = []
                for c in colorlabels_legend[labelname]["colors"]:
                    colorlines.append(Line2D([0], [0], color=c, lw=4))
                leg = ax.legend(
                    colorlines,
                    colorlabels_legend[labelname]["labels"],
                    bbox_to_anchor=(1.5 + 0.3 * i, 1.0),
                    title=labelname,
                )
                ax.add_artist(leg)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if colorlabels:
        maxr = R * 1.05 + width * len(colorlabels) + space * (len(colorlabels) - 1)
    elif sample_classes:
        maxr = R * 1.05 + width * len(sample_classes) + space * (len(sample_classes) - 1)
    else:
        maxr = R * 1.05
    ax.set_xlim(-maxr, maxr)
    ax.set_ylim(-maxr, maxr)
    return ax

def plot(
    Z2,
    fontsize=8,
    figsize=[100, 100],
    palette="gist_rainbow",
    addlabels=True,
    show=True,
    sample_classes=None,
    colorlabels=None,
    colorlabels_legend=None):
    """
    Drawing a radial dendrogram from a scipy dendrogram output.
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["svg.fonttype"] = "none"

    if colorlabels is not None:
        figsize = [100, 100]
    elif sample_classes is not None:
        figsize = [100, 100]

    fig, ax = plt.subplots(figsize=figsize)
    ax = radialTree(
        Z2,
        fontsize=fontsize,
        ax=ax,
        palette=palette,
        addlabels=addlabels,
        sample_classes=sample_classes,
        colorlabels=colorlabels,
        colorlabels_legend=colorlabels_legend,
    )

    if show:
        fig.show()
    else:
        return ax

def mat_plot(mat):
    # Take a matrix data instead of a dendrogram data, calculate dendrogram and draw a circular dendrogram
    pass

def pandas_plot(df):
    pass