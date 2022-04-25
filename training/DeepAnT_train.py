import torch
import numpy as np
from utils.evaluation import excess_mass, mass_volume, metric_calc, accuracy, precision, recall


class DeepAntTrainingPipeline():
    '''
    Training
    '''
    @staticmethod
    def train_epoch(model, loss_fn, train_dl, optimizer, DEVICE):
        loss_sum = 0.0
        for (X, Y, next_steps, next_labels) in train_dl:
            X = X.to(device=DEVICE, dtype=torch.float)
            next_step_seq = next_steps.squeeze().to(device=DEVICE, dtype=torch.float)
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
            for (X, Y, next_steps, next_labels) in test_dl:
                em, mv = self.deepant_emmv(model, X.to(DEVICE), next_steps.to(DEVICE), DEVICE=DEVICE)
                tmp_em += em
                tmp_mv += mv
            print("EMMV evaluation:")
            print(tmp_em / len(test_dl), tmp_mv / len(test_dl))
            total_em += tmp_em / len(test_dl)
            total_mv += tmp_mv / len(test_dl)
        print('Final results - EM: {0} MV: {1}'.format(total_em/len(tscv_dl_list), total_mv/len(tscv_dl_list)))

    def train_kdd99(self, train_dl, test_dl, model, optimizer, loss, num_epochs, DEVICE):
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch(model, loss, train_dl, optimizer, DEVICE)
            print('Epoch {0} loss: {1}'.format(epoch, epoch_loss))
        self.evaluate(model, test_dl, 1, DEVICE)

    def train_kdd99_full(self, train_dl_normal, train_dl_anomaly, test_dl_normal, test_dl_anomaly, model, optimizer, loss, num_epochs, DEVICE):
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch(model, loss, train_dl_normal, optimizer, DEVICE)
            epoch_loss += self.train_epoch(model, loss, train_dl_anomaly, optimizer, DEVICE)
            print('Epoch {0} loss: {1}'.format(epoch, epoch_loss/2))
        self.evaluate(model, test_dl_anomaly, 1, DEVICE)
        self.evaluate(model, test_dl_normal, 0, DEVICE)
    '''
    Evaluation
    '''
    @torch.no_grad()
    def evaluate(self, model, test_dl, label, DEVICE):
        model.eval()
        total_em = total_mv = total_acc = total_pre = total_rec = 0
        for (X, Y, next_steps, next_labels) in test_dl:
            seq_prediction = model(X.to(DEVICE)).detach().unsqueeze(dim=1)
            prediction = model.anomaly_detector(seq_prediction, next_steps.to(DEVICE), model.anomaly_threshold)
            true_positives, true_negatives, false_positives, false_negatives = metric_calc(prediction, next_labels, label)
            total_acc += accuracy(true_positives, true_negatives, next_labels)
            if (true_positives + false_positives) > 0:
                total_pre += precision(true_positives, false_positives)
            if (true_positives + false_negatives) > 0:
                total_rec += recall(true_positives, false_negatives)
            em, mv = self.deepant_emmv(model, X.to(DEVICE), next_steps.to(DEVICE), DEVICE=DEVICE)
            total_mv += mv
            total_em += em
        print("Acc: {0}, Pre: {1}, Rec: {2}".format(total_acc / len(test_dl), total_pre / len(test_dl), total_rec / len(test_dl)))
        print("EM: {0}, MV: {1}".format(total_em / len(test_dl), total_mv / len(test_dl)))

    # ***************************************************************************************
    # This method is an adaptation inspired by O'leary (2022) under the MIT open license.
    # Availability: https://github.com/christian-oleary/emmv
    # Note: the fundamental logic is not changed, but the pytorch implementation and customisation to support the
    # model associated with this training pipeline is novel.
    # ***************************************************************************************
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
        AUC_em, em, amax = excess_mass(t, t_max, volume_support, s_unif_total, anomaly_score, n_generated)
        AUC_mv, mv = mass_volume(axis_alpha, volume_support, s_unif_total, anomaly_score, n_generated)

        return np.mean(em), np.mean(mv)
