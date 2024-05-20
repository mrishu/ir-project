import numpy as np
from scipy import optimize


def MLE_lambda_EXP_T(sample_tfs, tau):
    sample_mean = np.sum(sample_tfs) / len(sample_tfs)
    MLE_lambda_EXP = 1 / sample_mean

    if tau == np.inf:
        return MLE_lambda_EXP

    if sample_mean >= tau / 2:
        return 0

    def LL(lam):
        return 1 / lam * tau / (np.exp(lam * tau) - 1) - sample_mean

    return optimize.newton(LL, 0)


def MLE_k_GLL_T(sample_tfs, tau):
    sample_log_mean = np.sum(np.log(1 + sample_tfs)) / len(sample_tfs)
    MLE_k_GLL = 1 / sample_log_mean

    if tau == np.inf:
        return MLE_k_GLL

    def LL(k):
        return 1 / k - k / (1 + tau)((1 + tau)**k - 1) - sample_log_mean

    return optimize.newton(LL, 0)
