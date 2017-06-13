import json
import matplotlib
import numpy as np
import sys


# REPORTING
def print_progress(string):
    sys.stdout.write('\r%s' % (string))
    sys.stdout.flush()


# JSON
def json_load(path):
    with open(path, 'r') as f:
        data_json = json.load(f)
    return data_json

def json_dump(var, path, indent=4):
    with open(path, 'wb') as f:
        json.dump(var, f, indent=indent, sort_keys=True, separators=(',', ': '))


# PLOTTING
def matplotlib_configure_as_notebook():
    # Matplotlib configuration for consistency with jupyter notebook
    # (where paper figures were originally produced)
    matplotlib.rcParams.update({
        'font.size': 10,
        'figure.subplot.bottom': 0.125,
        })

def tableau20(k):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)
    return tableau20[k]

def remove_chartjunk(ax):
    # remove top and side borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # change bottom border to dotted line
    ax.spines["bottom"].set_linestyle('dotted')
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["bottom"].set_color('black')
    ax.spines["bottom"].set_alpha(0.3)

    # remove ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off",
                   labelbottom="on", left="off", right="off", labelleft="on")

def plot_MSEs_to_axis(ax, alphas, MSEs, MSEs_stdev, do_confidence, labels, colors):
    ax.set_yscale('log')

    # Plot sets of MSE values vs alpha, possibly with confidence envelopes
    for MSE, stdev, label, color in zip(MSEs, MSEs_stdev, labels, colors):
        ax.plot(alphas, MSE, ls='solid', color=color, label=label)
        if do_confidence:
            num_stdevs = 1.0
            ax.fill_between(alphas, MSE-num_stdevs*stdev, MSE+num_stdevs*stdev, color=color, alpha=.2)

    ax.grid(b=True, which='major', linestyle='dotted', lw=0.5, color='black', alpha=0.3)
    ax.set_xlim((np.min(alphas), np.max(alphas)))

def save_plot(fig, savepath, bbox_extra_artists=None):
    fig.savefig(savepath + '.pdf', bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
    fig.savefig(savepath + '.png', bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
    print('Plot saved to %s.pdf and %s.png' % (savepath, savepath))
