import numpy as np
import torch
import random
from models.MADGAN import AnomalyDetector
from utils.evaluation import excess_mass, mass_volume
from utils.evaluation import accuracy, precision, recall, metric_calc


class MadGanTrainingPipeline():
    def set_seed(self, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def sample_Z(self, batch_size, seq_length, latent_dim):
        sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))
        return torch.Tensor(sample)

    '''
    Training
    '''
    def train_epoch(self, G, D, loss_fn, train_dl, G_optimizer, D_optimizer, seq_length, latent_dim, DEVICE,
                    normal_label=0, anomaly_label=1, epoch=0):
        G.train()
        D.train()
        g_loss_total = d_loss_real_total = d_loss_fake_total = 0

        for i, (X, Y) in enumerate(train_dl):
            bs = X.size(0)

            # Samples
            real_samples = X.to(DEVICE)
            latent_samples = self.sample_Z(bs, seq_length, latent_dim).to(DEVICE)
            fake_samples = G(latent_samples)

            # Labels
            real_labels = torch.full((bs, seq_length, 1), normal_label).float().to(DEVICE)
            fake_labels = torch.full((bs, seq_length, 1), anomaly_label).float().to(DEVICE)

            # Train D
            D_optimizer.zero_grad()
            real_d = D(real_samples)
            fake_d = D(fake_samples.detach())

            d_loss_real = loss_fn(real_d.view(-1), real_labels.view(-1))
            d_loss_fake = loss_fn(fake_d.view(-1), fake_labels.view(-1))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()

            D_optimizer.step()

            # train G
            G_optimizer.zero_grad()
            fake_d = D(fake_samples)
            g_loss = loss_fn(fake_d.view(-1), real_labels.view(-1))
            g_loss.backward()
            G_optimizer.step()

            # Save metrics
            g_loss_total += g_loss.item()
            d_loss_real_total += d_loss_real.item()
            d_loss_fake_total += d_loss_fake.item()
        D.zero_grad()
        G.zero_grad()
        print("Epoch {0}: G_loss: {1}, D_loss_real: {2}, D_loss_fake: {3}".format(epoch, g_loss_total / len(train_dl),
                    d_loss_real_total / len(train_dl), d_loss_fake_total / len(train_dl)))

    def train_financial(self, seq_length, latent_dim, tscv_dl_list, D, G, D_optim, G_optim, anomaly_threshold, loss_fn, random_seed, num_epochs, DEVICE) -> None:
        self.set_seed(random_seed)

        total_em = total_mv = 0
        ad = AnomalyDetector(discriminator=D, generator=G, latent_space_dim=latent_dim, anomaly_threshold=anomaly_threshold, DEVICE=DEVICE)
        for train_dl, test_dl in tscv_dl_list:
            for epoch in range(num_epochs):
                self.train_epoch(G, D, loss_fn, train_dl, G_optim, D_optim, seq_length, latent_dim, DEVICE,
                                 normal_label=0, anomaly_label=1, epoch=epoch)
            tmp_em = tmp_mv = 0
            for X, Y in test_dl:
                em, mv = self.emmv(ad, X.to(DEVICE), DEVICE=DEVICE)
                tmp_em += em
                tmp_mv += mv
            print("EM: {0}, MV: {1}".format(tmp_em / len(test_dl), tmp_mv / len(test_dl)))
            total_em += tmp_em / len(test_dl)
            total_mv += tmp_mv / len(test_dl)
        print('Final results - EM: {0} MV: {1}'.format(total_em / len(tscv_dl_list), total_mv / len(tscv_dl_list)))

    def train_kdd99(self, seq_length, latent_dim, train_dl, test_dl, D, G, D_optim, G_optim, anomaly_threshold, loss_fn, random_seed, num_epochs, DEVICE) -> None:
        self.set_seed(random_seed)

        for epoch in range(num_epochs):
            self.train_epoch(G, D, loss_fn, train_dl, G_optim, D_optim, seq_length, latent_dim, DEVICE, epoch=epoch)
        ad = AnomalyDetector(discriminator=D, generator=G, latent_space_dim=latent_dim, anomaly_threshold=anomaly_threshold, DEVICE=DEVICE)
        self.evaluate(ad, test_dl, label=1, DEVICE=DEVICE)

    '''
    Evaluation
    '''
    def evaluate(self, model, test_dl, label, DEVICE):
        total_em = total_mv = total_acc = total_pre = total_rec = 0
        for X, Y in test_dl:
            prediction = model.predict(X.to(DEVICE))
            true_positives, true_negatives, false_positives, false_negatives = metric_calc(prediction.view(-1, 1), Y.view(-1, 1), label)
            total_acc += accuracy(true_positives, true_negatives, Y)
            if (true_positives+false_positives) > 0:
                total_pre += precision(true_positives, false_positives)
            if (true_positives+false_negatives) > 0:
                total_rec += recall(true_positives, false_negatives)
            em, mv = self.emmv(model, X.to(DEVICE), DEVICE=DEVICE)
            total_mv += mv
            total_em += em
        print("Acc: {0}, Pre: {1}, Rec: {2}".format(total_acc/len(test_dl), total_pre/len(test_dl), total_rec/len(test_dl)))
        print("EM: {0}, MV: {1}".format(total_em/len(test_dl), total_mv/len(test_dl)))

    # ***************************************************************************************
    # This method is an adaptation of the original by O'leary (2022) under the MIT open license.
    # Availability: https://github.com/christian-oleary/emmv
    # Note: the fundamental logic is not changed, but the pytorch implementation and customisation to support the
    # model associated with this training pipeline is novel.
    # ***************************************************************************************
    def emmv(self, trained_model, x, n_generated=10000, alpha_min=0.9, alpha_max=0.999, t_max=0.9, DEVICE="cpu"):
        # Get limits and volume support.
        lim_inf = torch.min(x.view(-1, x.size(-1)), dim=0)[0]
        lim_sup = torch.max(x.view(-1, x.size(-1)), dim=0)[0]
        offset = 1e-60  # to prevent division by 0

        # Volume support
        volume_support = torch.prod(lim_sup - lim_inf).item() + offset

        # Determine EM and MV parameters
        t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
        axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)

        # Get anomaly scores
        anomaly_score = trained_model.predict(x).view(-1, 1).detach().cpu().numpy()

        reducer = 10
        reduced_n = int(n_generated / reducer)
        s_unif_list = []
        for i in range(reducer):
            unif = torch.rand(reduced_n, x.size(1), x.size(2)).to(DEVICE)
            m = lim_sup - lim_inf
            unif = unif * m
            unif = unif + lim_inf
            s_unif = trained_model.predict(unif).view(-1, 1).detach()
            s_unif_list.append(s_unif)
        s_unif_total = torch.cat(s_unif_list).cpu().numpy()

        # Get EM and MV scores
        AUC_em, em, amax = excess_mass(t, t_max, volume_support, s_unif_total, anomaly_score, n_generated)
        AUC_mv, mv = mass_volume(axis_alpha, volume_support, s_unif_total, anomaly_score, n_generated)

        return np.mean(em), np.mean(mv)
