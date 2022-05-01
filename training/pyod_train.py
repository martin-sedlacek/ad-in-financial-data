import torch
from utils.evaluation import excess_mass, mass_volume, metric_calc, accuracy, precision, recall
import numpy as np
import time


def train_kdd99(model, train_x, train_y, test_x, test_y):
    start = time.time()
    model.fit(train_x)
    end = time.time()
    print("Training time: {0}".format(end - start))
    start = time.time()
    prediction = model.predict(test_x)
    end = time.time()
    print("Prediction time: {0}".format(end - start))
    true_positives, true_negatives, false_positives, false_negatives = metric_calc(torch.tensor(prediction).unsqueeze(dim=1), torch.tensor(test_y).unsqueeze(dim=1), 1)
    acc = accuracy(true_positives, true_negatives, torch.tensor(test_y).unsqueeze(dim=1))
    pre = precision(true_positives, false_positives) if (true_positives + false_positives) > 0 else 0.0
    rec = recall(true_positives, false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    print("Acc: {0}, Pre: {1}, Rec: {2}".format(acc, pre, rec))
    em, mv = emmv(model, test_x)
    print("EM: {0}, MV: {1}".format(em, mv))


def train_financial(model, tscv_list):
    total_em = total_mv = 0
    for train_x, train_next_steps, test_x, test_next_steps in tscv_list:
        model.fit(train_x.squeeze())
        em, mv = emmv(model, test_x.squeeze())
        total_em += em
        total_mv += mv
        print(em, mv)
    print("EM: {0}, MV: {1}".format(total_em/len(tscv_list), total_mv/len(tscv_list)))


# ***************************************************************************************
# This method is provided by O'leary (2022) under the MIT open license.
# Small adjustments were made to fit the evaluation procedure.
# Availability: https://github.com/christian-oleary/emmv
# ***************************************************************************************
def emmv(trained_model, x, n_generated=10000, alpha_min=0.9, alpha_max=0.999, t_max=0.9):
    # Get limits and volume support.
    lim_inf = x.min(axis=0)
    lim_sup = x.max(axis=0)

    offset = 1e-60  # to prevent division by 0

    # Volume support
    volume_support = (lim_sup - lim_inf).prod() + offset

    # Determine EM and MV parameters
    t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
    axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)

    unif = np.random.uniform(lim_inf, lim_sup, size=(n_generated, x.shape[1]))

    # Get anomaly scores
    anomaly_score = trained_model.predict(x)
    s_unif = trained_model.predict(unif)

    # Get EM and MV scores
    AUC_em, em, amax = excess_mass(t, t_max, volume_support, s_unif, anomaly_score, n_generated)
    AUC_mv, mv = mass_volume(axis_alpha, volume_support, s_unif, anomaly_score, n_generated)

    return np.mean(em), np.mean(mv)
