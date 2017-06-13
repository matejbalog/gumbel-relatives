import argparse
import matplotlib.pylab as plt
import numpy as np

from scipy.special import digamma

from tricks import EULER
from utils import json_load, matplotlib_configure_as_notebook, tableau20, save_plot

""" Figure 3a
    Sample size M required to reach a given MSE using Gumbel and Exponential
    trick estimators of ln(Z), using samples from A* sampling on a Robust
    Bayesian Regression task.
"""

def main(args_dict):
    # Extract configuration from command line arguments
    MK = args_dict['MK']
    Kmin = args_dict['Kmin']

    # Load data
    data = json_load('data/astar_rbr_MK%d.json' % (MK))
    lnZ = data['lnZ']
    MAPs = np.array(data['MAPs'])
    print('Loaded %d MAP samples from A* sampling' % (len(MAPs)))

    # Estimate MSE of lnZ estimators from Gumbel and Exponential tricks
    MSEs_Gumb = []
    MSEs_Expo = []
    Ms = xrange(1, MK / Kmin)
    for M in Ms:
        # Computation with M samples, repeated K >= Kmin times with a new set every time
        K = MK / M
        myMAPs = np.reshape(MAPs[:(K*M)], (K, M))
        
        # Compute unbiased estimators of ln(Z)
        lnZ_Gumb = np.mean(myMAPs, axis=1)
        lnZ_Expo = EULER - np.log(np.mean(np.exp(- myMAPs), axis=1)) - (np.log(M) - digamma(M))
        
        # Save MSE estimates
        MSEs_Gumb.append(np.mean((lnZ_Gumb - lnZ) ** 2))
        MSEs_Expo.append(np.mean((lnZ_Expo - lnZ) ** 2))
            
    # Set up plot
    matplotlib_configure_as_notebook()
    fig, ax = plt.subplots(1, 1, facecolor='w', figsize=(4.25, 3.25))
    ax.set_xscale('log')
    ax.set_xlabel('desired MSE (lower to the right)')
    ax.set_ylabel('required number of samples $M$')
    ax.grid(b=True, which='both', linestyle='dotted', lw=0.5, color='black', alpha=0.3)
    
    # Plot MSEs
    ax.plot(MSEs_Gumb, Ms, color=tableau20(0), label='Gumbel')
    ax.plot(MSEs_Expo, Ms, color=tableau20(2), label='Exponential')

    # Finalize plot
    ax.set_xlim((1e-2, 2))
    ax.invert_xaxis()
    lgd = ax.legend(loc='upper left')
    save_plot(fig, 'figures/fig3a', (lgd,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--MK', default=100000, type=int, help='total number of samples available')
    parser.add_argument('--Kmin', default=1000, type=int, help='minimum number of repetitions to estimate MSE of an estimator')
    args_dict = vars(parser.parse_args())
    main(args_dict)
