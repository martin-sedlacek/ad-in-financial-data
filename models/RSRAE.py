import torch
from torch import nn
import torch.nn.functional as F


class RSRLayer(nn.Module):
    def __init__(self, d: int, D: int):
        super().__init__()
        self.d = d
        self.D = D
        self.A = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(d, D)))

    def forward(self, z):
        # z is the output from the encoder
        z_hat = self.A @ z.view(z.size(0), self.D, 1)
        return z_hat.squeeze(2)


class RSRAutoEncoder(nn.Module):
    def __init__(self, input_dim, d, D):
        super().__init__()
        # Put your encoder network here, remember about the output D-dimension
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, D)
        )

        self.rsr = RSRLayer(d, D)

        # Put your decoder network here, rembember about the input d-dimension
        self.decoder = nn.Sequential(
            nn.Linear(d, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        enc = self.encoder(x)  # obtain the embedding from the encoder
        latent = self.rsr(enc)  # RSR manifold
        #F.normalize(latent, p=2)
        dec = self.decoder(latent)  # obtain the representation in the input space
        return enc, dec, latent, self.rsr.A
