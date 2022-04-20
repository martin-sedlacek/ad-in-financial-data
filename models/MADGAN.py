import torch
from torch import nn
from pathlib import Path
from typing import Optional, Union
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, latent_space_dim: int, hidden_units: int, output_dim: int, n_lstm_layers: int = 2) -> None:
        super().__init__()
        self.latent_space_dim = latent_space_dim
        self.hidden_units = hidden_units
        self.n_lstm_layers = n_lstm_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_size=self.latent_space_dim,
                            hidden_size=self.hidden_units,
                            num_layers=self.n_lstm_layers,
                            batch_first=True,
                            dropout=.1)

        self.linear = nn.Linear(in_features=self.hidden_units,
                                out_features=self.output_dim)

        nn.init.trunc_normal_(self.linear.bias)
        nn.init.trunc_normal_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_output, _ = self.lstm(x)
        return self.linear(rnn_output)

    def save(self, fpath: Union[Path, str]) -> None:
        chkp = {
            "config": {
                "latent_space_dim": self.latent_space_dim,
                "hidden_units": self.hidden_units,
                "n_lstm_layers": self.n_lstm_layers,
                "output_dim": self.output_dim
            },
            "weights": self.state_dict(),
        }
        torch.save(chkp, fpath)

    @classmethod
    def from_pretrained(
            cls,
            fpath: Union[Path, str],
            map_location: Optional[torch.device] = None) -> "Generator":
        chkp = torch.load(fpath, map_location=map_location)
        model = cls(**chkp.pop("config"))
        model.load_state_dict(chkp.pop("weights"))
        model.eval()
        return model


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_units, n_lstm_layers: int = 2, add_batch_mean: bool = True) -> None:
        super().__init__()
        self.add_batch_mean = add_batch_mean
        self.hidden_units = hidden_units
        self.input_dim = input_dim
        self.n_lstm_layers = n_lstm_layers

        extra_features = self.hidden_units if self.add_batch_mean else 0
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_units + extra_features,
                            num_layers=self.n_lstm_layers,
                            batch_first=True,
                            dropout=.1)

        self.linear = nn.Linear(in_features=self.hidden_units + extra_features,
                                out_features=1)
        nn.init.trunc_normal_(self.linear.bias)
        nn.init.trunc_normal_(self.linear.weight)

        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.add_batch_mean:
            bs = x.size(0)
            batch_mean = x.mean(0, keepdim=True).repeat(bs, 1, 1)
            x = torch.cat([x, batch_mean], dim=-1)

        rnn_output, _ = self.lstm(x)
        return self.activation(self.linear(rnn_output))

    def save(self, fpath: Union[Path, str]) -> None:
        chkp = {
            "config": {
                "add_batch_mean": self.add_batch_mean,
                "hidden_units": self.hidden_units,
                "input_dim": self.input_dim,
                "n_lstm_layers": self.n_lstm_layers
            },
            "weights": self.state_dict(),
        }
        torch.save(chkp, fpath)

    @classmethod
    def from_pretrained(
            cls,
            fpath: Union[Path, str],
            map_location: Optional[torch.device] = None) -> "Discriminator":
        chkp = torch.load(fpath, map_location=map_location)
        model = cls(**chkp.pop("config"))
        model.load_state_dict(chkp.pop("weights"))
        model.eval()
        return model


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

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        return (self.predict_proba(tensor) > self.threshold).int()

    def predict_proba(self, tensor: torch.Tensor) -> torch.Tensor:
        discriminator_score = self.compute_anomaly_score(tensor)
        discriminator_score *= 1 - self.res_weight
        reconstruction_loss = self.compute_reconstruction_loss(tensor)
        reconstruction_loss *= self.res_weight
        return discriminator_score #+ reconstruction_loss

    def compute_anomaly_score(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            discriminator_score = self.discriminator(tensor)
        return discriminator_score

    def compute_reconstruction_loss(self,
                                    tensor: torch.Tensor) -> torch.Tensor:
        best_reconstruct = self._generate_best_reconstruction(tensor)
        return (best_reconstruct - tensor).abs().sum(dim=(1, 2))

    def _generate_best_reconstruction(self, tensor: torch.Tensor) -> None:
        # The goal of this function is to find the corresponding latent space for the given
        # input and then generate the best possible reconstruction.
        max_iters = 10

        Z = torch.empty(
            (tensor.size(0), tensor.size(1), self.latent_space_dim),
            requires_grad=True, device=self.DEVICE)
        nn.init.normal_(Z, std=0.05)

        optimizer = torch.optim.RMSprop(params=[Z], lr=0.1)
        loss_fn = nn.MSELoss(reduction="none")
        normalized_target = F.normalize(tensor, dim=1, p=2)

        for _ in range(max_iters):
            optimizer.zero_grad()
            generated_samples = self.generator(Z)
            normalized_input = F.normalize(generated_samples, dim=1, p=2)
            reconstruction_error = loss_fn(normalized_input,
                                           normalized_target).sum(dim=(0, 1, 2))
            reconstruction_error.backward()
            optimizer.step()

        with torch.no_grad():
            best_reconstruct = self.generator(Z)
        return best_reconstruct
