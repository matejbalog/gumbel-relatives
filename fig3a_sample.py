import argparse
import numpy as np

# Required script files from A* sampling (https://github.com/cmaddis/astar-sampling)
import astar
import robustbayesregr

from Gumbel import EULER
from utils import print_progress, json_dump


def main(args_dict):
    # Extract configuration
    MK = args_dict['MK']

    # Construct Robust Bayesian regression model
    sigma = 2 * np.ones(1)
    bounder = robustbayesregr.Bounder()
    splitter = robustbayesregr.Splitter()    
    proposal = robustbayesregr.IsotropicGaussian(1, sigma)
    np.random.seed(0)
    x, y = robustbayesregr.generate_data(1000)
    target = robustbayesregr.CauchyRegression(x, y, sigma)

    # Obtain MK samples (and their corresponding MAP values) using A* sampling implementation
    samples = np.empty((MK)).squeeze()
    MAPs = []
    for i in range(MK):
        stream = astar.astar_sampling_iterator(target, proposal, bounder, splitter)
        X, G = stream.next()
        samples[i] = X
        MAPs.append(G[0] - EULER)
        if i % 1 == 0:
            print_progress('Sampled %d / %d' % (i+1, MK))
    print('')
    lnZ = float(np.log(target.z()))
    
    # Dump true ln(Z) and MAP values to JSON file
    data = {'lnZ': lnZ, 'MAPs': MAPs}
    savepath = 'data/astar_rbr_MK%d.json' % (MK)
    json_dump(data, savepath, indent=None)
    print('Saved %d samples to %s' % (len(MAPs), savepath))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--MK', default=100000, type=int, help='total number of samples available')
    args_dict = vars(parser.parse_args())
    main(args_dict)
