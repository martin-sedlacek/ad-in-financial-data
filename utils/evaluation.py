import numpy as np
from sklearn.metrics import auc


# ***************************************************************************************
# this method was taken directly from the code associated with Goix (2016) without modification
# Availability: https://github.com/ngoix/EMMV_benchmarks; https://arxiv.org/abs/1607.01152
# ***************************************************************************************
def excess_mass(t, t_max, volume_support, s_unif, s_X, n_generated):
    EM_t = np.zeros(t.shape[0])
    n_samples = s_X.shape[0]
    s_X_unique = np.unique(s_X)
    EM_t[0] = 1.

    for u in s_X_unique:
        EM_t = np.maximum(EM_t, 1. / n_samples * (s_X > u).sum() - t * (s_unif > u).sum() / n_generated * volume_support)
    amax = np.argmax(EM_t <= t_max) + 1
    if amax == 1:
        amax = -1 # failed to achieve t_max
    AUC = auc(t[:amax], EM_t[:amax])
    return AUC, EM_t, amax


# ***************************************************************************************
# this method was taken directly from the code associated with Goix (2016) without modification
# Availability: https://github.com/ngoix/EMMV_benchmarks; https://arxiv.org/abs/1607.01152
# ***************************************************************************************
def mass_volume(axis_alpha, volume_support, s_unif, s_X, n_generated):
    n_samples = s_X.shape[0]
    s_X_argsort = s_X.argsort()
    mass = 0
    cpt = 0
    u = s_X[s_X_argsort[-1]]
    mv = np.zeros(axis_alpha.shape[0])
    for i in range(axis_alpha.shape[0]):
        while mass < axis_alpha[i]:
            cpt += 1
            u = s_X[s_X_argsort[-cpt]]
            mass = 1. / n_samples * cpt  # sum(s_X > u)
        mv[i] = float((s_unif >= float(u)).sum()) / n_generated * volume_support
    return auc(axis_alpha, mv), mv


def metric_calc(y_hat, y, desired_val):
    true_positives = true_negatives = false_positives = false_negatives = 0
    for i in range(y.size(0)):
        for j in range(y.size(1)):
            label = int(y[i][j].item())
            pred = int(y_hat[i][j].item())
            if label == pred and desired_val == label:
                true_positives += 1
            elif label == pred and desired_val != label:
                true_negatives += 1
            elif label != pred and desired_val == label:
                false_negatives += 1
            elif label != pred and desired_val != label:
                false_positives += 1
    return true_positives, true_negatives, false_positives, false_negatives


def accuracy(true_positives, true_negatives, y):
    return (true_positives+true_negatives) / (y.size(0)*y.size(1))


def precision(true_positives, false_positives):
    return true_positives / (true_positives+false_positives)


def recall(true_positives, false_negatives):
    return true_positives / (true_positives+false_negatives)



