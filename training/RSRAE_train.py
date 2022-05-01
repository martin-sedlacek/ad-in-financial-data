import torch
from utils.evaluation import excess_mass, mass_volume, metric_calc, accuracy, precision, recall
import numpy as np
import time


def train_epoch(model, train_dl, L2p_loss, RSRLoss, optimizer, scheduler, DEVICE, dynamic_lr=True):
    model.train()
    loss_sum = 0.0
    for X, Y in train_dl:
        x = X.view(X.size(0), -1).to(DEVICE)

        enc, dec, latent, A = model(x)
        rec_loss = L2p_loss(torch.sigmoid(dec), x)
        rsr_loss = RSRLoss(enc, A)
        loss = rec_loss + rsr_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_sum += loss.item()
    if dynamic_lr: scheduler.step()
    return loss_sum / len(train_dl)


def train_kdd99(model, train_dl, test_dl, num_epochs, L2p_loss, RSRLoss, optimizer, scheduler, anomaly_threshold, DEVICE):
    start = time.time()
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_dl, L2p_loss, RSRLoss, optimizer, scheduler, DEVICE=DEVICE, dynamic_lr=False) #!!! enable lr
        print("Epoch {0} loss: {1}".format(epoch, loss))
    end = time.time()
    print("Training time: {0}".format(end - start))
    evaluate(model, test_dl, anomaly_threshold, DEVICE=DEVICE)


def train_financial(model, tscv_dl_list, num_epochs, L2p_loss, RSRLoss, optimizer, scheduler, anomaly_threshold, DEVICE):
    total_em = total_mv = 0
    for train_dl, test_dl in tscv_dl_list:
        model.train()
        for epoch in range(num_epochs):
            loss = train_epoch(model, train_dl, L2p_loss, RSRLoss, optimizer, scheduler, DEVICE=DEVICE, dynamic_lr=False)
            print("Epoch {0} loss: {1}".format(epoch, loss))
        model.eval()
        tmp_em = tmp_mv = 0
        for X, Y in test_dl:
            em, mv = emmv(model, X.detach(), anomaly_threshold)
            tmp_mv += mv
            tmp_em += em
        total_em += tmp_em / len(test_dl)
        total_mv += tmp_mv / len(test_dl)
        print("EM: {0}, MV: {1}".format(tmp_em / len(test_dl), tmp_mv / len(test_dl)))
    print('Final results - EM: {0} MV: {1}'.format(total_em / len(tscv_dl_list), total_mv / len(tscv_dl_list)))


def evaluate(trained_model, test_dl, anomaly_threshold, DEVICE):
    trained_model.eval()
    total_em = total_mv = total_acc = total_pre = total_rec = 0
    total_time = 0
    for X, Y in test_dl:
        x = X.view(X.size(0), -1).detach()
        y = Y.view(Y.size(0), -1).detach()

        start = time.time()
        enc, x_hat, latent, A = trained_model(x)
        x = X.detach().numpy()
        x_hat = x_hat.view(X.size(0), -1).detach()
        x_hat = x_hat.view(X.size(0), X.size(1), -1).numpy()

        cosine_similarity = np.sum(x_hat * x, -1) / (np.linalg.norm(x_hat, axis=-1) + 0.000001) / (np.linalg.norm(x, axis=-1) + 0.000001)
        prediction = torch.tensor(cosine_similarity < anomaly_threshold).float()
        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(dim=1)
        end = time.time()
        total_time += end - start
        #prediction = prediction.view(-1, 1)

        true_positives, true_negatives, false_positives, false_negatives = metric_calc(prediction, y, 1)
        total_acc += accuracy(true_positives, true_negatives, y)
        if (true_positives + false_positives) > 0:
            total_pre += precision(true_positives, false_positives)
        if (true_positives + false_negatives) > 0:
            total_rec += recall(true_positives, false_negatives)

        em, mv = emmv(trained_model, X.detach(), anomaly_threshold)
        total_mv += mv
        total_em += em
    print("Detection time: {0}".format(total_time))
    print("Acc: {0}, Pre: {1}, Rec: {2}".format(total_acc / len(test_dl), total_pre / len(test_dl), total_rec / len(test_dl)))
    print("EM: {0}, MV: {1}".format(total_em / len(test_dl), total_mv / len(test_dl)))


def emmv(trained_model, x, anomaly_threshold, n_generated=10000, alpha_min=0.9, alpha_max=0.999, t_max=0.9, DEVICE="cpu"):
    # Get limits and volume support.
    lim_inf = torch.min(x.view(-1, x.size(-1)), dim=0)[0]
    lim_sup = torch.max(x.view(-1, x.size(-1)), dim=0)[0]
    offset = 1e-60  # to prevent division by 0

    bs = x.size(0)
    seq_len = x.size(1)
    num_features = x.size(2)
    x = x.view(x.size(0), -1).detach()

    # Volume support
    volume_support = torch.prod(lim_sup - lim_inf).item() + offset

    # Determine EM and MV parameters
    t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
    axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)

    # Get anomaly scores
    enc, x_hat, latent, A = trained_model(x)
    x_np = x.numpy().squeeze()
    x_hat = x_hat.view(x.size(0), -1).detach()
    x_hat = x_hat.view(x.size(0), x.size(1), -1).numpy().squeeze()
    cosine_similarity = np.sum(x_hat * x_np, -1) / (np.linalg.norm(x_hat, axis=-1) + 0.000001) / (np.linalg.norm(x_np, axis=-1) + 0.000001)
    anomaly_score = torch.tensor(cosine_similarity < anomaly_threshold).float().cpu().numpy() #.unsqueeze(dim=1)

    reducer = 10
    reduced_n = int(n_generated / reducer)
    s_unif_list = []
    for i in range(reducer):
        unif = torch.rand(reduced_n, seq_len, num_features).to(DEVICE)
        m = lim_sup - lim_inf
        unif = unif * m
        unif = unif + lim_inf
        unif = unif.view(unif.size(0), -1).detach()

        enc, unif_hat, latent, A = trained_model(unif)
        unif_np = unif.numpy().squeeze()
        unif_hat = unif_hat.view(int(n_generated / reducer), -1).detach().numpy().squeeze()
        cosine_similarity = np.sum(unif_hat * unif_np, -1) / (np.linalg.norm(unif_hat, axis=-1) + 0.000001) / (np.linalg.norm(unif_np, axis=-1) + 0.000001)
        s_unif = torch.tensor(cosine_similarity < anomaly_threshold).float()#.unsqueeze(dim=1)
        s_unif_list.append(s_unif)
    s_unif_total = torch.cat(s_unif_list).cpu().numpy()

    # Get EM and MV scores
    AUC_em, em, amax = excess_mass(t, t_max, volume_support, s_unif_total, anomaly_score, n_generated)
    AUC_mv, mv = mass_volume(axis_alpha, volume_support, s_unif_total, anomaly_score, n_generated)

    return np.mean(em), np.mean(mv)
