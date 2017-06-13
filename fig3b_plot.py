import argparse
import matplotlib.pylab as plt
import numpy as np

from tricks import MAPs_to_estimator_MSE_vs_alpha
from utils import json_load, matplotlib_configure_as_notebook, plot_MSEs_to_axis, save_plot

""" Figure 3b
    MSE of ln(Z) estimators for different values of alpha, using M=100 samples
    from the approximate MAP algorithm discussed in Section 5.2 of the paper,
    with different error bounds delta.
"""

def main(args_dict):
    # Extract configuration from command line arguments
    MK = np.array(args_dict['MK'])
    M = 100
    K = MK / M
    print('M = %d; K = %d' % (M, K))
    x_type = args_dict['x_type']
    deltas = args_dict['deltas']
    do_confidence = args_dict['confidence']
    
    # Load data from JSON files generated by (non-public) Matlab code
    jsons = [json_load('data/bandits_normal_delta%s_MK%d.json' % (delta, MK)) for delta in deltas]
    lnZs = np.array([json['lnZ'] for json in jsons])
    MAPs = np.array([json['MAPs_ttest'] for json in jsons])
    
    # Estimate estimator MSEs for the various tricks (as specified by alphas)
    alphas = np.linspace(-0.2, 1.5, 100)
    MSEs, MSEs_stdev = MAPs_to_estimator_MSE_vs_alpha(1, MAPs, lnZs, alphas, K)

    # Set up plot
    matplotlib_configure_as_notebook()
    fig, ax = plt.subplots(1, 1, facecolor='w', figsize=(4.25, 3.25))
    ax.set_xlabel('trick parameter $\\alpha$')
    ax.set_ylabel('MSE of estimator of $\ln Z$')
    
    # Plot the MSEs
    labels = ['$\\delta = %g$' % (delta) for delta in deltas]
    colors = [plt.cm.plasma((np.log10(delta) - (-3)) / (0 - (-3))) for delta in deltas]
    plot_MSEs_to_axis(ax, alphas, MSEs, MSEs_stdev, do_confidence, labels, colors)

    # Finalize plot
    for vertical in [0.0, 1.0]:
        ax.axvline(vertical, color='black', linestyle='dashed', alpha=.7)
    ax.annotate('Gumbel trick', xy=(0.0, 0.0052), rotation=90, horizontalalignment='right', verticalalignment='bottom')
    ax.annotate('Exponential trick', xy=(1.0, 0.0052), rotation=90, horizontalalignment='right', verticalalignment='bottom')
    lgd = ax.legend(loc='upper center')
    ax.set_ylim((5*1e-3, 5*1e-2))
    save_plot(fig, 'figures/fig3b', bbox_extra_artists=(lgd,))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--MK', default=100000, type=int, help='total number of samples (value of MxK)')
    parser.add_argument('--x_type', default='normal', help='x_type from {normal, lognormal, uniform, bernoulli}', action='store')
    parser.add_argument('--deltas', default=[0.001, 0.01, 0.1], nargs='+', type=float, help='error bounds delta to plot')
    parser.add_argument('--confidence', help='show confidence envelopes', action='store_true')
    args_dict = vars(parser.parse_args())
    main(args_dict)
