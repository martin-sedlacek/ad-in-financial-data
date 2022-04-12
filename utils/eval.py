import numpy as np
from sklearn.metrics import auc
import torch


def excess_mass(t, t_max, volume_support, s_unif, s_X, n_generated):
    EM_t = np.zeros(t.shape[0])
    n_samples = s_X.shape[0]
    s_X_unique = np.unique(s_X)
    EM_t[0] = 1.

    for u in s_X_unique:
        EM_t = np.maximum(EM_t, 1. / n_samples * (s_X > u).sum() -
                        t * (s_unif > u).sum() / n_generated
                        * volume_support)
    amax = np.argmax(EM_t <= t_max) + 1
    if amax == 1:
        amax = -1 # failed to achieve t_max
    AUC = auc(t[:amax], EM_t[:amax])
    return AUC, EM_t, amax


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


def torch_emmv_scores(trained_model, x, scoring_func=None, n_generated=10000, alpha_min=0.9, alpha_max=0.999,
                      t_max=0.9):
    # Get limits and volume support.
    lim_inf = torch.min(x.view(-1, 6), dim=0)[0]
    lim_sup = torch.max(x.view(-1, 6), dim=0)[0]
    offset = 1e-60  # to prevent division by 0

    # Volume support
    volume_support = torch.prod(lim_sup - lim_inf).item() + offset

    # Determine EM and MV parameters
    t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
    axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)

    unif = torch.rand(n_generated, x.size(1), x.size(2))
    m = lim_sup - lim_inf
    unif = unif * m
    unif = unif + lim_inf

    # Get anomaly scores
    anomaly_score = scoring_func(trained_model, x).view(-1, 1).detach().numpy()
    s_unif = scoring_func(trained_model, unif).view(-1, 1).detach().numpy()

    # Get EM and MV scores
    AUC_em, em, amax = excess_mass(t, t_max, volume_support, s_unif, anomaly_score, n_generated)
    AUC_mv, mv = mass_volume(axis_alpha, volume_support, s_unif, anomaly_score, n_generated)

    # Return a dataframe containing EMMV information
    scores = {
        'em': np.mean(em),
        'mv': np.mean(mv),
    }
    return scores


def accuracy(y_hat, y):  #y_hat is a matrix; 2nd dimension stores prediction scores for each class.
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(torch.sum(cmp))
