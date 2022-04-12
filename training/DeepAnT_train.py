import numpy as np
import torch
from training.training_utils import BaseTrainingPipeline
from utils import eval


class DeepAntTrainingPipeline(BaseTrainingPipeline):
    def anomaly_detector(self, prediction_seq, ground_truth_seq, anm_det_thr):
        # calculate Euclidean between actual seq and predicted seq
        # ist = np.linalg.norm(ground_truth_seq - prediction_seq)
        dist = np.absolute(ground_truth_seq - prediction_seq).mean(axis=1)
        return (dist > anm_det_thr).astype(float)

    def torch_scoring_function(self, model, data):
        return model(data)

    def evaluate(self, model, test_dl):
        model.eval()
        total_acc = total_em = total_mv = 0
        ctr = 0
        for X, Y, P in test_dl:
            pred = model(X).detach()
            anomaly_predict = torch.tensor(self.anomaly_detector(pred.numpy(), P.numpy(), model.anomaly_threshold))
            anomaly_label = Y.squeeze()
            acc = eval.accuracy(anomaly_predict, anomaly_label)
            total_acc += acc
            ctr += 1
            scores = eval.torch_emmv_scores(model, X, scoring_func=self.torch_scoring_function)
            total_em += scores['em']
            total_mv += scores['mv']
        print(total_acc / ctr, total_em / ctr, total_mv / ctr)
        return total_acc / ctr, total_em / ctr, total_mv / ctr
