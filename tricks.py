import numpy as np

from scipy.special import gamma, digamma, polygamma

EULER = 0.5772156649


# Parameter estimation: analytically computed estimator variances and MSEs
# for the Gumbel and Exponential trick estimators of Z and ln(Z), constructed
# from M observations (MAP solutions)
def Z_Gumbel_var(M):
    return np.power(gamma(1.0-2.0/M), M) * np.exp(- 2*EULER) - np.power(gamma(1.0-1.0/M), 2*M) * np.exp(-2*EULER)

def Z_Gumbel_MSE(M):
    return np.power(gamma(1.0-2.0/M), M) * np.exp(- 2*EULER) - 2*np.power(gamma(1.0-1.0/M), M) * np.exp(-EULER) + 1.0

def Z_Exponential_var(M):
    return 1.0 * (M**2) / (((M-1)**2) * (M-2))

def Z_Exponential_MSE(M):
    return 1.0 * (M+2) / ((M-1) * (M-2))

def lnZ_Gumbel_MSE(M):
    return (np.pi ** 2) / (6 * M)

def lnZ_Exponential_var(M):
    return polygamma(1, M)

def lnZ_Exponential_MSE(M):
    return (np.log(M) - digamma(M)) ** 2 + lnZ_Exponential_var(M)


# Estimating estimator MSEs by sampling
def MAPs_to_estimator_MSE_vs_alpha(n, MAPs, lnZs, alphas_in, K):
    """ Estimator MSE vs alpha (parameter specifying Frechet, Weibull, or
        Exponential trick), estimated from sets of MAP samples.

        Calling this with n = 1 corresponds to the full-rank setting, while
        n equal to the number of variables corresponds to unary perturbations.
    """

    # Find and mask indices where alpha is zero (Gumbel trick requires special
    # treatment to avoid division by zero)
    idx_zero = np.abs(alphas_in) < 0.001
    alphas = np.array(alphas_in)
    alphas[idx_zero] = 0.001

    # Reshape samples (each estimator is based on M observations, and each
    # estimator is constructed K times to assess its MSE)
    num_models = np.shape(MAPs)[0]
    M = np.shape(MAPs)[1] / K
    MAPs = MAPs[:, :(M*K)]                                  # (num_models, M*K)
    MAPs = np.reshape(MAPs, (num_models, K, M, 1))          # (num_models, K, M, 1)

    # Compute ln(Z) estimates by averaging in exponential space
    MAPs_alphas = MAPs * alphas                                                         # (num_models, K, M, alphas)
    mean_exp = np.mean(np.exp(- MAPs_alphas), axis=2)                                   # (num_models, K, alphas)
    lnZ_hat = n*np.log(gamma(1.0+alphas))/alphas + n*EULER - np.log(mean_exp) / alphas  # (num_models, K, alphas)
    
    # Fix zero alphas (the Gumbel trick)
    lnZ_hat[:, :, idx_zero] = np.mean(MAPs, axis=2)

    # Estimate estimator MSEs
    SEs = (lnZ_hat - np.reshape(lnZs, (-1, 1, 1))) ** 2     # (num_models, K, alphas)
    MSEs = np.mean(SEs, axis=1)                             # (num_models, alphas)
    stdev = np.std(SEs, axis=1) / np.sqrt(K)                # (num_models, alphas)
    return MSEs, stdev
