import torch
import numpy as np

class DeepAnT_CNN(torch.nn.Module):
    def __init__(self, LOOKBACK_SIZE, DIMENSION, anomaly_threshold):
        super(DeepAnT_CNN, self).__init__()
        self.anomaly_threshold = anomaly_threshold
        self.conv1d_1_layer = torch.nn.Conv1d(in_channels=LOOKBACK_SIZE, out_channels=64, kernel_size=1)
        self.relu_1_layer = torch.nn.ReLU()
        self.maxpooling_1_layer = torch.nn.MaxPool1d(kernel_size=2)
        self.conv1d_2_layer = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.relu_2_layer = torch.nn.ReLU()
        self.maxpooling_2_layer = torch.nn.MaxPool1d(kernel_size=2)
        self.flatten_layer = torch.nn.Flatten()
        self.dense_1_layer = torch.nn.Linear(64, 40)
        self.relu_3_layer = torch.nn.ReLU()
        self.dropout_layer = torch.nn.Dropout(p=0.25)
        self.dense_2_layer = torch.nn.Linear(40, DIMENSION)

    def forward(self, x):
        x = self.conv1d_1_layer(x)
        x = self.relu_1_layer(x)
        x = self.maxpooling_1_layer(x)
        x = self.conv1d_2_layer(x)
        x = self.relu_2_layer(x)
        x = self.maxpooling_2_layer(x)
        x = self.flatten_layer(x)
        x = self.dense_1_layer(x)
        x = self.relu_3_layer(x)
        x = self.dropout_layer(x)
        x = self.dense_2_layer(x)
        return x

    @staticmethod
    def anomaly_detector(prediction_seq, ground_truth_seq, anm_det_thr):
        # calculate Euclidean between actual seq and predicted seq
        # ist = np.linalg.norm(ground_truth_seq - prediction_seq)
        dist = np.absolute(ground_truth_seq - prediction_seq).mean(axis=prediction_seq.ndim-1)
        return (dist > anm_det_thr).astype(float)


class DeepAnT_LSTM(torch.nn.Module):
    def __init__(self, LOOKBACK_SIZE, DIMENSION, anomaly_threshold):
        super(DeepAnT_LSTM, self).__init__()
        self.anomaly_threshold = anomaly_threshold
        self.lstm_1_layer = torch.nn.LSTM(DIMENSION, 128, 2)
        self.dropout_1_layer = torch.nn.Dropout(p=0.2)
        self.lstm_2_layer = torch.nn.LSTM(128, 64, 2)
        self.dropout_2_layer = torch.nn.Dropout(p=0.2)
        self.linear_layer = torch.nn.Linear(64, DIMENSION)

    def forward(self, x):
        x, (_, _) = self.lstm_1_layer(x)
        x = self.dropout_1_layer(x)
        x, (_, _) = self.lstm_2_layer(x)
        x = self.dropout_2_layer(x)
        return self.linear_layer(x)

    @staticmethod
    def anomaly_detector(prediction_seq, ground_truth_seq, anm_det_thr):
        # calculate Euclidean between actual seq and predicted seq
        # ist = np.linalg.norm(ground_truth_seq - prediction_seq)
        dist = np.absolute(ground_truth_seq - prediction_seq).mean(axis=prediction_seq.ndim-1)
        return (dist > anm_det_thr).astype(float)
