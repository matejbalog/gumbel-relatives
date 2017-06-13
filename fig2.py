import argparse
import matplotlib.pylab as plt
import numpy as np

from matplotlib import gridspec
from scipy.special import gamma

from tricks import Z_Gumbel_MSE, Z_Exponential_MSE, lnZ_Gumbel_MSE, lnZ_Exponential_MSE
from utils import print_progress, matplotlib_configure_as_notebook, remove_chartjunk, plot_MSEs_to_axis, save_plot

""" Figure 2
    MSE of estimators of Z (left) and ln(Z) (right) stemming from Frechet
    (-1/2 < alpha < 0), Gumbel (alpha = 0) and Weibull tricks (alpha > 0).

    Observation 1: With full-rank perturbations, the structure of the
    underlying model does not matter. The MAP solutions follow a distribution
    that only depends on the true value of the normalizing constant Z.

    Observation 2: Thanks to scaling, we can assume WLOG when evaluating
    estimators that Z = 1 and lnZ = 0.
"""

def estimate_MSE_vs_alpha(transform, Ms, alphas, K):
    # Without loss of generality
    Z = 1
    tZ = transform(Z)
    
    # Estimate MSEs by constructing estimators K times
    MSEs = np.empty((len(Ms), len(alphas)))
    MSEs_stdev = np.empty((len(Ms), len(alphas)))
    for Mi, M in enumerate(Ms):
        # Compute means (K x alphas) in a loop, as otherwise
        # this runs out of memory with K = 100,000.
        means = np.empty((K, len(alphas)))
        for ai, alpha in enumerate(alphas):
            Ws = np.power(np.random.exponential(1.0, size=(K, M)), alpha)   # (K, M)
            means[:, ai] = np.mean(Ws, axis=1)
            print_progress('M = %d: done %.0f%%' % (M, 100.0 * (ai+1) / len(alphas)))
        print('')

        g = np.power(gamma(1.0 + alphas), 1.0 / alphas)         # (alphas)
        tZ_hats = transform(g * np.power(means, -1.0/alphas))   # (K, alphas)
        SEs = (tZ_hats - tZ) ** 2                               # (K, alphas)
        MSEs[Mi] = np.mean(SEs, axis=0)                         # (alphas)
        MSEs_stdev[Mi] = np.std(SEs, axis=0) / np.sqrt(K)       # (alphas)

    return MSEs, MSEs_stdev


def main(args_dict):
    # Extract configuration from command line arguments
    Ms = np.array(args_dict['Ms'])
    alphas = np.linspace(args_dict['alpha_min'], args_dict['alpha_max'], args_dict['alpha_num'])
    K = args_dict['K']
    do_confidence = args_dict['confidence']

    # Estimate MSEs by sampling
    print('Estimating MSE of estimators of Z...')
    MSEs_Z, MSE_stdevs_Z = estimate_MSE_vs_alpha(lambda x: x, Ms, alphas, K)
    print('Estimating MSE of estimators of ln(Z)...')
    MSEs_lnZ, MSE_stdevs_lnZ = estimate_MSE_vs_alpha(np.log, Ms, alphas, K)

    # Set up plot
    matplotlib_configure_as_notebook()
    fig = plt.figure(facecolor='w', figsize=(8.25, 3.25))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.5])
    ax = [plt.subplot(gs[0]), plt.subplot(gs[2]), plt.subplot(gs[1])]

    ax[0].set_xlabel('$\\alpha$')
    ax[2].set_xlabel('$\\alpha$')
    ax[0].set_ylabel('MSE of estimators of $Z$, in units of $Z^2$')
    ax[2].set_ylabel('MSE of estimators of $\ln Z$, in units of $1$')
    
    colors = [plt.cm.plasma(0.8 - 1.0 * i / len(Ms)) for i in xrange(len(Ms))]

    # Gumbel (alpha=0) and Exponential (alpha=1) tricks can be handled analytically
    legend_Gumbel = 'Gumbel trick\n($\\alpha=0$, theoretical)'
    legend_Exponential = 'Exponential trick\n($\\alpha=1$, theoretical)'
    ax[0].scatter(np.zeros(len(Ms)), Z_Gumbel_MSE(Ms), marker='o', color=colors, label=legend_Gumbel)
    ax[0].scatter(np.ones(len(Ms)), Z_Exponential_MSE(Ms), marker='^', color=colors, label=legend_Exponential)
    ax[2].scatter(np.zeros(len(Ms)), lnZ_Gumbel_MSE(Ms), marker='o', color=colors, label=legend_Gumbel)
    ax[2].scatter(np.ones(len(Ms)), lnZ_Exponential_MSE(Ms), marker='^', color=colors, label=legend_Exponential)

    # Remaining tricks MSE were estimated by sampling
    labels = ['$M=%d$' % (M) for M in Ms]
    plot_MSEs_to_axis(ax[0], alphas, MSEs_Z, MSE_stdevs_Z, do_confidence, labels, colors)
    plot_MSEs_to_axis(ax[2], alphas, MSEs_lnZ, MSE_stdevs_lnZ, do_confidence, labels, colors)

    # Finalize plot
    ax[0].set_ylim((5*1e-3, 10))
    ax[2].set_ylim((5*1e-3, 10))
    handles, labels = ax[0].get_legend_handles_labels()
    remove_chartjunk(ax[1])
    ax[1].spines["bottom"].set_visible(False)
    ax[1].tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="off", left="off", right="off", labelleft="off")
    ax[1].legend(handles, labels, frameon=False, loc='upper center', bbox_to_anchor=[0.44, 1.05])
    plt.tight_layout()
    save_plot(fig, 'figures/fig2_K%d' % (K))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ms', default=[4, 8, 16, 32, 64, 128], nargs='+', type=int, help='sample sizes M to evaluate')
    parser.add_argument('--alpha_min', default=-0.15, type=float, help='minimum alpha to evaluate')
    parser.add_argument('--alpha_max', default=+2.00, type=float, help='maximum alpha to evaluate')
    parser.add_argument('--alpha_num', default=100, type=int, help='number of alphas to evaluate')
    parser.add_argument('--K', default=100000, type=int, help='number of estimator constructions to assess variance')
    parser.add_argument('--confidence', help='show confidence envelopes', action='store_true')
    args_dict = vars(parser.parse_args())
    main(args_dict)
