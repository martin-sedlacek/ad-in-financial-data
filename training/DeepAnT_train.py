import torch
from utils import evaluation
import numpy as np


class DeepAntTrainingPipeline():
    @staticmethod
    def train_epoch(model, loss_fn, train_dl, optimizer, DEVICE):
        loss_sum = 0.0
        for X, Y in train_dl:
            X = X.to(device=DEVICE, dtype=torch.float)
            next_step_seq = Y.squeeze().to(device=DEVICE, dtype=torch.float)
            model.train()
            pred_next = model(X).squeeze().to(device=DEVICE, dtype=torch.float)
            loss = loss_fn(next_step_seq, pred_next)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.item()
        return float(loss_sum / len(train_dl))

    def train(self, tscv_dl_list, model, optimizer, loss, num_epochs, DEVICE):
        total_em = total_mv = 0
        for train_dl, test_dl in tscv_dl_list:
            for epoch in range(num_epochs):
                epoch_loss = self.train_epoch(model, loss, train_dl, optimizer, DEVICE)
                print('Epoch {0} loss: {1}'.format(epoch, epoch_loss))

            tmp_em = tmp_mv = 0
            for X, Y in test_dl:
                scores = self.deepant_emmv(model, X.to(DEVICE), Y.to(DEVICE), DEVICE=DEVICE)
                tmp_em += scores['em']
                tmp_mv += scores['mv']
            print("EMMV evaluation:")
            print(tmp_em / len(test_dl), tmp_mv / len(test_dl))
            total_em += tmp_em / len(test_dl)
            total_mv += tmp_mv / len(test_dl)
        print('Final results - EM: {0} MV: {1}'.format(total_em/len(tscv_dl_list), total_mv/len(tscv_dl_list)))

    @staticmethod
    def deepant_emmv(trained_model, pre_x, x, n_generated=10000, alpha_min=0.9, alpha_max=0.999, t_max=0.9, DEVICE="cpu"):
        def scoring_function(model, pre_seq, seq):
            seq_pred = model(pre_seq).detach()
            prediction = model.anomaly_detector(seq_pred, seq, model.anomaly_threshold)
            return prediction

        offset = 1e-60  # to prevent division by 0

        # Get limits and volume support.
        pre_x_lim_inf = torch.min(pre_x.view(-1, pre_x.size(-1)), dim=0)[0]
        pre_x_lim_sup = torch.max(pre_x.view(-1, pre_x.size(-1)), dim=0)[0]
        x_lim_inf = torch.min(x.view(-1, x.size(-1)), dim=0)[0]
        x_lim_sup = torch.max(x.view(-1, x.size(-1)), dim=0)[0]
        volume_support = torch.prod(x_lim_sup - x_lim_inf).item() + offset
        t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
        axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)

        reducer = 10
        reduced_n = int(n_generated / reducer)
        s_unif_list = []
        for i in range(reducer):
            pre_unif = torch.rand(reduced_n, pre_x.size(1), pre_x.size(2)).to(DEVICE)
            m = pre_x_lim_sup - pre_x_lim_inf
            pre_unif = pre_unif * m
            pre_unif = pre_unif + pre_x_lim_inf

            unif = torch.rand(reduced_n, x.size(1), x.size(2)).to(DEVICE)
            m = x_lim_sup - x_lim_inf
            unif = unif * m
            unif = unif + x_lim_inf

            s_unif = scoring_function(trained_model, pre_unif, unif).view(-1, 1).detach()
            s_unif_list.append(s_unif)
        s_unif_total = torch.cat(s_unif_list).cpu().numpy()

        # Get anomaly scores
        anomaly_score = scoring_function(trained_model, pre_x, x).view(-1, 1).detach().cpu().numpy()

        # Get EM and MV scores
        AUC_em, em, amax = evaluation.excess_mass(t, t_max, volume_support, s_unif_total, anomaly_score, n_generated)
        AUC_mv, mv = evaluation.mass_volume(axis_alpha, volume_support, s_unif_total, anomaly_score, n_generated)

        # Return a dataframe containing EMMV information
        scores = {
            'em': np.mean(em),
            'mv': np.mean(mv),
        }
        return scores
