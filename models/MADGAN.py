import torch
from torch import nn
import torch.nn.functional as F


# ***************************************************************************************
# The following MAD-GAN implementation was inspired by the source code from Guillem96 (2022) with varying alterations
# Availability: https://github.com/Guillem96/madgan-pytorch
# ***************************************************************************************
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=0.2)
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=self.output_dim)

        nn.init.trunc_normal_(self.linear.bias)
        nn.init.trunc_normal_(self.linear.weight)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers=2):
        super().__init__()
        self.hidden_units = hidden_size
        self.input_dim = input_dim
        self.num_layers = num_layers

        # batch_first=True,
        # extra_feature
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_units, num_layers=self.num_layers, dropout=0.2)
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)
        self.activation = nn.Sigmoid()

        nn.init.trunc_normal_(self.linear.bias)
        nn.init.trunc_normal_(self.linear.weight)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.activation(self.linear(out))


class AnomalyDetector(object):
    def __init__(self,
                 *,
                 discriminator: nn.Module,
                 generator: nn.Module,
                 latent_space_dim: int,
                 res_weight: float = .2,
                 anomaly_threshold: float = 1.0,
                 DEVICE = "cpu") -> None:
        self.discriminator = discriminator.to(DEVICE)
        self.generator = generator.to(DEVICE)
        self.threshold = anomaly_threshold
        self.latent_space_dim = latent_space_dim
        self.res_weight = res_weight
        self.DEVICE = DEVICE

    def predict(self, tensor):
        return (self.predict_proba(tensor) > self.threshold).int()

    def predict_proba(self, tensor):
        discriminator_score = self.compute_anomaly_score(tensor)
        discriminator_score *= 1 - self.res_weight
        reconstruction_loss = self.compute_reconstruction_loss(tensor)
        reconstruction_loss *= self.res_weight
        return discriminator_score + reconstruction_loss

    def compute_anomaly_score(self, tensor):
        with torch.no_grad():
            discriminator_score = self.discriminator(tensor)
        return discriminator_score

    def compute_reconstruction_loss(self, tensor):
        best_reconstruct = self._generate_best_reconstruction(tensor)
        return (best_reconstruct - tensor).abs().sum(dim=(1, 2))

    def _generate_best_reconstruction(self, tensor: torch.Tensor):
        # The goal of this function is to find the corresponding latent space for the given
        # input and then generate the best possible reconstruction.
        max_iters = 10

        Z = torch.empty((tensor.size(0), tensor.size(1), self.latent_space_dim), requires_grad=True, device=self.DEVICE)
        nn.init.normal_(Z, std=0.05)

        optimizer = torch.optim.RMSprop(params=[Z], lr=0.1)
        loss_fn = nn.MSELoss(reduction="none")
        normalized_target = F.normalize(tensor, dim=1, p=2)

        for _ in range(max_iters):
            optimizer.zero_grad()
            generated_samples = self.generator(Z)
            normalized_input = F.normalize(generated_samples, dim=1, p=2)
            reconstruction_error = loss_fn(normalized_input, normalized_target).sum(dim=(0, 1, 2))
            reconstruction_error.backward()
            optimizer.step()

        with torch.no_grad():
            best_reconstruct = self.generator(Z)
        return best_reconstruct
