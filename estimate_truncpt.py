import numpy as np
from scipy.stats import bootstrap


def exp_lambda_mle(data):
    return len(data) / np.sum(data)


def gll_k_mle(data):
    return len(data) / np.sum(np.log(1 + data))


def truncpt_EXP_T(normalized_tfs_sample, delta):
    if delta == 0:
        return np.inf

    normalized_tfs_sample = np.array(normalized_tfs_sample)
    lam_mle = exp_lambda_mle(normalized_tfs_sample)
    truncpt = - np.log(delta) / lam_mle

    # Check if lam is a good estimate using parametric bootstrap.
    # If not, then return inf i.e. do not truncate.
    confidence_interval = bootstrap((normalized_tfs_sample,), exp_lambda_mle).confidence_interval
    if confidence_interval.low <= lam_mle <= confidence_interval.high:
        return truncpt
    return np.inf


def truncpt_GLL_T(normalized_tfs_sample, delta):
    if delta == 0:
        return np.inf

    normalized_tfs_sample = np.array(normalized_tfs_sample)
    k_mle = gll_k_mle(normalized_tfs_sample)
    truncpt = delta**(-1 / k_mle) - 1

    # Check if truncpt_mle_estimate is a good estimate using parametric bootstrap.
    # If not, then return inf i.e. do not truncate.
    confidence_interval = bootstrap((normalized_tfs_sample,), gll_k_mle).confidence_interval
    if confidence_interval.low <= k_mle <= confidence_interval.high:
        return truncpt
    return np.inf
