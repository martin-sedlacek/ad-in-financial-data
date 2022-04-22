from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear1 = nn.Linear(in_features=self.input_dim, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=self.output_dim)

    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = F.dropout(out, 0.2)
        out = self.linear2(out)
        out = F.relu(out)
        out = F.dropout(out, 0.2)
        out = self.linear3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear1 = nn.Linear(in_features=self.input_dim, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=128)
        self.linear4 = nn.Linear(in_features=128, out_features=self.output_dim)

    def forward(self, x):
        out = self.linear1(x)
        out = F.leaky_relu(out, 0.1)
        out = F.dropout(out, 0.1)
        out = self.linear2(out)
        out = F.leaky_relu(out, 0.1)
        out = F.dropout(out, 0.1)
        out = self.linear3(out)
        out = F.leaky_relu(out, 0.1)
        out = F.dropout(out, 0.1)
        out = self.linear4(out)
        out = F.sigmoid(out)
        return out


