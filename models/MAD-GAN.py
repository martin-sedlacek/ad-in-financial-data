import numpy as np
import torch
from torch import nn
import random
from pathlib import Path
from typing import Optional, Union


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# --- sample Z from latent space --- #
def sample_Z(batch_size, seq_length, latent_dim, use_time=False, use_noisy_time=False):
    sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))
    if use_time:
        print('WARNING: use_time has different semantics')
        sample[:, :, 0] = np.linspace(0, 1.0 / seq_length, num=seq_length)
    return torch.Tensor(sample)


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
