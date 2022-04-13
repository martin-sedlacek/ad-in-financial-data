import numpy as np
import torch
from utils import evaluation


class DeepAntTrainingPipeline():
    @staticmethod
    def anomaly_detector(prediction_seq, ground_truth_seq, anm_det_thr):
        # calculate Euclidean between actual seq and predicted seq
        # ist = np.linalg.norm(ground_truth_seq - prediction_seq)
        dist = np.absolute(ground_truth_seq - prediction_seq).mean(axis=1)
        return (dist > anm_det_thr).astype(float)

    @staticmethod
    def torch_scoring_function(model, data):
        return model(data)

    @staticmethod
    def train_epoch(model, loss_fn, optimizer, train_dl):
        loss_sum = 0.0
        ctr = 0
        for x_batch, y_batch, next_batch in train_dl:
            model.train()
            pred_next = model(x_batch)
            loss = loss_fn(next_batch, pred_next)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train = loss.item()
            loss_sum += loss_train
            ctr += 1
        return float(loss_sum / ctr)

    def train(self, model, loss, optimizer, train_dl, test_dl, num_epochs, per_epoch_evaluation=False):
        for epoch in range(num_epochs):
            epoch_train_loss = self.train_epoch(model, loss, optimizer, train_dl)
            print("Training Loss: {0} - Epoch: {1}".format(epoch_train_loss, epoch + 1))
            if per_epoch_evaluation:
                test_acc, test_em, test_mv = self.evaluate(model, test_dl)
                print("Acc: {0}, EM: {1}, MV: {2} - Epoch: {3}".format(test_acc, test_em, test_mv, epoch + 1))

    @torch.no_grad()
    def evaluate(self, model, test_dl):
        model.eval()
        total_acc = total_em = total_mv = 0
        ctr = 0
        for X, Y, P in test_dl:
            pred = model(X).detach()
            anomaly_predict = torch.tensor(self.anomaly_detector(pred.numpy(), P.numpy(), model.anomaly_threshold))
            anomaly_label = Y.squeeze()
            acc = evaluation.accuracy(anomaly_predict, anomaly_label)
            total_acc += acc
            ctr += 1
            scores = evaluation.torch_emmv_scores(model, X, scoring_func=self.torch_scoring_function)
            total_em += scores['em']
            total_mv += scores['mv']
        return total_acc / ctr, total_em / ctr, total_mv / ctr
