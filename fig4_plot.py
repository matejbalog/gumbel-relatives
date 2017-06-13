import argparse
import matplotlib.pylab as plt
import numpy as np

from matplotlib import gridspec
from scipy.special import gamma

from libdai import load_libdai_results
from tricks import MAPs_to_estimator_MSE_vs_alpha
from utils import matplotlib_configure_as_notebook, remove_chartjunk, plot_MSEs_to_axis, save_plot

""" Figure 4
    MSEs of U(alpha) as estimators of ln(Z) on 10x10 attractive (left and
    middle subplot) and mixed (right subplot) spin glass model with different
    coupling strengths C.
"""

def plot_MSEs_and_optima_to_axis(ax, MSEs, MSEs_stdev, xlabel, Cs, alphas, MIN, do_confidence):
    # Axes
    ax.set_xlabel('trick parameter $\\alpha$\n%s' % (xlabel))
    
    # Plot MSEs
    labels = ['$C=%1.2g$' % (C) for C in Cs]
    colors = [plt.cm.plasma(0.8 * (C - np.min(Cs)) / (np.max(Cs) - np.min(Cs))) for C in Cs]
    plot_MSEs_to_axis(ax, alphas, MSEs, MSEs_stdev, do_confidence, labels, colors)
    
    # Plot optima
    minima_from = MIN
    alphas_mask = alphas[alphas >= minima_from]
    MSEs_mask = MSEs[:, alphas >= minima_from]
    for Mi in xrange(len(MSEs) - 1, -1, -1):
        idx = np.argmin(MSEs_mask[Mi])
        pos_x = alphas_mask[idx]
        pos_y = MSEs_mask[Mi][idx]
        ax.scatter([pos_x], [pos_y], color=colors[Mi], marker='*')
        idx_zero = list(alphas_mask).index(0.0)
        assert idx_zero >= 0
        percentage_samples = 100.0 * (1.0 - (MSEs_mask[Mi][idx] / MSEs_mask[Mi][idx_zero]))
        alignment = 'right'
        if (xlabel=='attractive, JCT') and (Mi==3):
            alignment = 'left'
        if (xlabel=='mixed, JCT') and (Mi==1):
            alignment = 'left'
        ax.text(pos_x, pos_y, '$%.0f\%%$' % (percentage_samples), color=colors[Mi], horizontalalignment=alignment)
        
    # Plot vertical lines
    for vertical in [2*MIN, MIN, 0]:
        ax.axvline(vertical, color='black', linestyle='dashed', alpha=.7)
        
    # Modify labels on horizontal axis
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = '-0.1'
    ax.set_xticklabels(['', '-0.05', '0', '0.05', ''])


def load_and_plot_to_axis(ax, potentials_type, MAP_solvers_file, MAP_solver, args_dict):
    # Extract model parameters
    topology = 'grid'
    A = args_dict['A']
    B = args_dict['B']
    n = A * B
    R = args_dict['R']
    f = args_dict['f']
    Cs = args_dict['Cs']

    # Extract computation parameters
    M = args_dict['M']
    K = args_dict['K']

    # Load data
    lnZs = np.empty(len(Cs))
    MAPs = np.empty((len(Cs), M*K))
    for ci, c in enumerate(Cs):
        data_json = load_libdai_results(topology, n, A, B, R, f, c, potentials_type, M*K, MAP_solvers_file)
        lnZs[ci] = data_json['lnZ']
        MAPs[ci] = data_json['MAPs_unary_%s' % (MAP_solver)]

    # alphas to plot
    MIN = -0.5/np.sqrt(n)
    alphas_spaced = np.linspace(-0.1, 0.1, 41)
    alphas_fixed = [MIN, 0.0]
    alphas = np.array(sorted(list(alphas_spaced) + alphas_fixed))

    # Compute MSE and plot
    MSEs, MSEs_stdev = MAPs_to_estimator_MSE_vs_alpha(n, MAPs, lnZs, alphas, K)
    xlabel = '%s, %s' % (potentials_type, MAP_solver.replace('JT', 'JCT'))
    plot_MSEs_and_optima_to_axis(ax, MSEs, MSEs_stdev, xlabel, Cs, alphas, MIN, args_dict['confidence'])


def main(args_dict):
    # Set up plot
    matplotlib_configure_as_notebook()
    fig = plt.figure(facecolor='w', figsize=(9.3, 3.6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.0, 1.0, 0.70, 1.0])
    gs.update(wspace=0.025, hspace=0.05)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[2], sharey=ax0)
    ax2 = fig.add_subplot(gs[1], sharey=ax0)
    ax3 = fig.add_subplot(gs[3], sharey=ax0)
    ax = [ax0, ax1, ax2, ax3]

    # Vertical axes
    ax[0].set_ylabel('MSE of estimators of $\ln Z$')
    ax[2].tick_params(axis="both", which="both", labelleft="off")
    ax[3].tick_params(axis="both", which="both", labelleft="off")

    # Load and plot MSEs
    load_and_plot_to_axis(ax[0], 'attractive', 'JT_BP', 'JT', args_dict)
    load_and_plot_to_axis(ax[2], 'attractive', 'JT_BP', 'BP', args_dict)
    load_and_plot_to_axis(ax[3], 'mixed', 'JT', 'JT', args_dict)

    # Legend
    handles, labels = ax[0].get_legend_handles_labels()
    remove_chartjunk(ax[1])
    ax[1].spines["bottom"].set_visible(False)
    ax[1].tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="off", left="off", right="off", labelleft="off")
    ax[1].legend(handles, labels, frameon=False, loc='upper center', bbox_to_anchor=[0.44, 1.05])
    save_plot(fig, 'figures/fig4')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--A', default=10, type=int, help='spin glass model width')
    parser.add_argument('--B', default=10, type=int, help='spin glass model height')
    parser.add_argument('--R', default=2, type=int, help='number of different labels a variable can take')
    parser.add_argument('--f', default=1.0, type=float, help='range of unary potentials')
    parser.add_argument('--Cs', default=(3.0 * np.array([9, 7, 5, 3, 1, 0]) / 9), nargs='+', type=float, help='coupling strengths')
    parser.add_argument('--M', default=100, type=int, help='sample size on which each estimator of ln(Z) is based')
    parser.add_argument('--K', default=1000, type=int, help='number of estimator constructions to assess variance')
    parser.add_argument('--confidence', help='show confidence envelopes', action='store_true')
    args_dict = vars(parser.parse_args())
    main(args_dict)
