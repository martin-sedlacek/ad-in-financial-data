import torch
from utils import evaluation
import numpy as np


class DeepAntTrainingPipeline():
    @staticmethod
    def train_epoch(model, loss_fn, optimizer, train_dl, DEVICE):
        loss_sum = 0.0
        ctr = 0
        for x_batch, y_batch, next_batch, next_batch_labels in train_dl:
            x_batch = x_batch.to(DEVICE)
            next_batch = next_batch.to(DEVICE)
            model.train()
            pred_next = model(x_batch).to(DEVICE)
            loss = loss_fn(next_batch, pred_next)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train = loss.item()
            loss_sum += loss_train
            ctr += 1
        return float(loss_sum / ctr)

    def train(self, model, loss, optimizer, train_dl, test_dl, num_epochs, DEVICE, per_epoch_evaluation=False):
        model = model.to(DEVICE)
        for epoch in range(num_epochs):
            epoch_train_loss = self.train_epoch(model, loss, optimizer, train_dl, DEVICE)
            print("Training Loss: {0} - Epoch: {1}".format(epoch_train_loss, epoch + 1))
            if per_epoch_evaluation:
                test_acc, test_em, test_mv = self.evaluate(model, test_dl, DEVICE)
                print("Acc: {0}, EM: {1}, MV: {2} - Epoch: {3}".format(test_acc, test_em, test_mv, epoch + 1))

    @staticmethod
    def deepant_emmv(trained_model, pre_x, x, scoring_func=None, n_generated=10000, alpha_min=0.9, alpha_max=0.999,t_max=0.9):
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

        # Generate pre_unif
        lim_inf = torch.min(pre_x.view(-1, 6), dim=0)[0]
        lim_sup = torch.max(pre_x.view(-1, 6), dim=0)[0]
        volume_support = torch.prod(lim_sup - lim_inf).item() + offset
        t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
        axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)
        pre_unif = torch.rand(n_generated, pre_x.size(1), pre_x.size(2))
        m = lim_sup - lim_inf
        pre_unif = pre_unif * m
        pre_unif = pre_unif + lim_inf

        # Get anomaly scores
        anomaly_score = scoring_func(trained_model, pre_x, x).flatten()
        s_unif = scoring_func(trained_model, pre_unif, unif).flatten()

        # Get EM and MV scores
        AUC_em, em, amax = evaluation.excess_mass(t, t_max, volume_support, s_unif, anomaly_score, n_generated)
        AUC_mv, mv = evaluation.mass_volume(axis_alpha, volume_support, s_unif, anomaly_score, n_generated)

        # Return a dataframe containing EMMV information
        scores = {
            'em': np.mean(em),
            'mv': np.mean(mv),
        }
        return scores

    @torch.no_grad()
    def evaluate(self, model, test_dl):
        total_em = total_mv = total_acc = total_precision = total_recall = 0
        model.eval()
        model = model.cpu()
        for X, Y, P, PL in test_dl:
            seq_prediction = model(X).detach()
            label_prediction = torch.tensor(model.anomaly_detector(seq_prediction.numpy(), P.numpy(), model.anomaly_threshold))

            true_positives, true_negatives, false_positives, false_negatives = evaluation.metric_calc(label_prediction, PL)
            acc = (true_positives + true_negatives) / (PL.size(0) * PL.size(1))
            total_acc += acc
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
                total_precision += precision
            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
                total_recall += recall

            scores = self.deepant_emmv(model, X, P, scoring_func=self.scoring_function)
            print(scores)

            total_em += scores['em']
            total_mv += scores['mv']
        print(total_em / len(test_dl), total_mv / len(test_dl))
        print(total_acc / len(test_dl), total_precision / len(test_dl), total_recall / len(test_dl))
        return total_em / len(test_dl), total_mv / len(test_dl)

    @staticmethod
    def scoring_function(model, pre_seq, seq):
        seq_pred = model(pre_seq).detach()
        prediction = model.anomaly_detector(seq_pred.numpy(), seq.numpy(), model.anomaly_threshold)
        return prediction
