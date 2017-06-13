import argparse
import matplotlib.pylab as plt
import numpy as np

from tricks import Z_Gumbel_MSE, Z_Gumbel_var, Z_Exponential_MSE, Z_Exponential_var, lnZ_Gumbel_MSE, lnZ_Exponential_MSE, lnZ_Exponential_var
from utils import tableau20, matplotlib_configure_as_notebook, save_plot

""" Figure 1
    Analytically computed MSE and variance of Gumbel and Exponential trick
    estimators of Z (left subplot) and ln(Z) (right subplot), using
    different sample sizes M.
"""

def main(args_dict):
    # Set up plot
    matplotlib_configure_as_notebook()
    fig, ax = plt.subplots(1, 2, facecolor='w', figsize=(9.25, 3.25))
    
    # Estimating Z
    Ms = np.arange(3, args_dict['M']+1)

    ax[0].set_xlabel('number of samples $M$')
    ax[0].set_ylabel('MSE of $\hat{Z}$, in units of $Z^2$')
    ax[0].set_xlim((np.min(Ms), np.max(Ms)))
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].grid(b=True, which='major', linestyle='dotted', lw=.5, color='black', alpha=0.5)

    ax[0].plot(Ms, Z_Gumbel_MSE(Ms), linestyle='-', color=tableau20(0), label='Gumbel: MSE')
    ax[0].plot(Ms, Z_Gumbel_var(Ms), linestyle='dashed', color=tableau20(0), label='Gumbel: var')
    ax[0].plot(Ms, Z_Exponential_MSE(Ms), linestyle='-', color=tableau20(2), label='Exponential: MSE')
    ax[0].plot(Ms, Z_Exponential_var(Ms), linestyle='dashed', color=tableau20(2), label='Exponential: var')

    # Estimating ln Z
    Ms = np.arange(1, args_dict['M']+1)

    ax[1].set_xlabel('number of samples $M$')
    ax[1].set_ylabel('MSE of $\widehat{\ln Z}$, in units of $1$')
    ax[1].set_xlim((np.min(Ms), np.max(Ms)))
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].grid(b=True, which='major', linestyle='dotted', lw=0.5, color='black', alpha=0.5)

    ax[1].plot(Ms, lnZ_Gumbel_MSE(Ms), linestyle='-', color=tableau20(0), label='Gumbel: MSE')
    ax[1].plot(Ms, lnZ_Exponential_MSE(Ms), linestyle='-', color=tableau20(2), label='Exponential: MSE')
    ax[1].plot(Ms, lnZ_Exponential_var(Ms), linestyle='dashed', color=tableau20(2), label='Exponential: var')

    # Finalize plot
    lgd0 = ax[0].legend()
    lgd1 = ax[1].legend()
    plt.tight_layout()
    save_plot(fig, 'figures/fig1', (lgd0, lgd1,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', default=10000, type=int, help='maximum sample size M to consider')
    args_dict = vars(parser.parse_args())
    main(args_dict)
