import torch


def anomaly_detector(prediction_seq, ground_truth_seq, anm_det_thr):
    # calculate Euclidean between actual seq and predicted seq
    dist = torch.sqrt(torch.square(ground_truth_seq - prediction_seq)).mean(dim=prediction_seq.dim() - 1)
    return (dist > anm_det_thr).to(dtype=torch.float32)


# ***************************************************************************************
# The following DeepAnT_CNN implementation was inspired by the source code from Bmonikraj (2020) with alterations.
# The class was re-written to smoothly integrate to the training pipeline in this project. Note that the LSTM
# version below was written from scratch. The idea to use LSTM is from the original paper by Munir et. al (2019).
# Availability: https://github.com/bmonikraj/medium-ds-unsupervised-anomaly-detection-deepant-lstmae
# ***************************************************************************************
class DeepAnT_CNN(torch.nn.Module):
    def __init__(self, lookback_dim, output_dim, kernel_size, hidden_dim, num_channels, anomaly_threshold):
        super(DeepAnT_CNN, self).__init__()
        self.anomaly_detector = anomaly_detector
        self.anomaly_threshold = anomaly_threshold
        self.conv1d_1_layer = torch.nn.Conv1d(in_channels=lookback_dim, out_channels=num_channels, kernel_size=kernel_size)
        self.relu_1_layer = torch.nn.ReLU()
        self.maxpooling_1_layer = torch.nn.MaxPool1d(kernel_size=2)
        self.conv1d_2_layer = torch.nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size)
        self.relu_2_layer = torch.nn.ReLU()
        self.maxpooling_2_layer = torch.nn.MaxPool1d(kernel_size=2)
        self.flatten_layer = torch.nn.Flatten()
        self.dense_1_layer = torch.nn.Linear(hidden_dim, 40) #64
        self.relu_3_layer = torch.nn.ReLU()
        self.dropout_layer = torch.nn.Dropout(p=0.25)
        self.dense_2_layer = torch.nn.Linear(40, output_dim)

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


class DeepAnT_LSTM(torch.nn.Module):
    def __init__(self, lookback_dim, hidden_dim, num_layers, anomaly_threshold):
        super(DeepAnT_LSTM, self).__init__()
        self.anomaly_detector = anomaly_detector
        self.anomaly_threshold = anomaly_threshold
        self.lstm_1_layer = torch.nn.LSTM(lookback_dim, hidden_dim, num_layers)
        self.dropout_1_layer = torch.nn.Dropout(p=0.2)
        self.lstm_2_layer = torch.nn.LSTM(hidden_dim, 64, num_layers)
        self.dropout_2_layer = torch.nn.Dropout(p=0.2)
        self.linear_layer = torch.nn.Linear(64, lookback_dim)

    def forward(self, x):
        x, (_, _) = self.lstm_1_layer(x)
        x = self.dropout_1_layer(x)
        x, (_, _) = self.lstm_2_layer(x)
        x = self.dropout_2_layer(x)
        return self.linear_layer(x)

