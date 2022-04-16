import numpy as np
import torch
import random
from typing import Dict
from pathlib import Path


class MadGanTrainingPipeline():
    def set_seed(self, seed: int = 0) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # --- sample Z from latent space --- #
    def sample_Z(self, batch_size, seq_length, latent_dim, use_time=False):
        sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))
        if use_time:
            print('WARNING: use_time has different semantics')
            sample[:, :, 0] = np.linspace(0, 1.0 / seq_length, num=seq_length)
        return torch.Tensor(sample)

    @torch.no_grad()
    def evaluate(self, G, D, loss_fn, real_dl, seq_length, latent_dim, DEVICE, normal_label: int = 0, anomaly_label: int = 1) -> Dict[str, float]:
        metric_accum = {
            "D_loss": 0,
            "G_acc": 0,
            "D_acc": 0
        }
        batch_count = 0
        for X, Y, P, PL in real_dl:
            bs = X.size(0)

            # Samples
            real_samples = X.to(DEVICE)
            latent_samples = self.sample_Z(bs, seq_length, latent_dim).to(DEVICE)
            fake_samples = G(latent_samples)

            # Labels
            real_labels = Y.to(DEVICE)
            fake_labels = torch.zeros(bs, seq_length, 1).to(DEVICE)
            all_labels = torch.cat([real_labels, fake_labels])

            # Try to classify the real and generated samples
            real_d = D(real_samples)
            fake_d = D(fake_samples.detach())
            all_d = torch.cat([real_d, fake_d]).to(DEVICE)

            # Discriminator tries to identify the true nature of each sample (anomaly or normal)
            d_real_loss = loss_fn(real_d.view(-1), real_labels.view(-1))
            d_fake_loss = loss_fn(fake_d.view(-1), fake_labels.view(-1))
            d_loss = d_real_loss + d_fake_loss

            discriminator_acc = ((all_d > .5) == all_labels).float()
            discriminator_acc = discriminator_acc.sum().div(bs * seq_length)

            generator_acc = ((fake_d > .5) == real_labels).float()
            generator_acc = generator_acc.sum().div(bs * seq_length)

            metric_accum["D_loss"] += d_loss.item()
            metric_accum["D_acc"] += discriminator_acc.item()
            metric_accum["G_acc"] += generator_acc.item()
            batch_count += 1
        for key in metric_accum.keys():
            metric_accum[key] = metric_accum[key] / batch_count
        print("Evaluation metrics:", metric_accum)
        return metric_accum

    def train_epoch(self, G, D, loss_fn, real_dl, G_optimizer, D_optimizer, seq_length, latent_dim, DEVICE,
                    normal_label: int = 0, anomaly_label: int = 1, epoch: int = 0, log_every: int = 30) -> None:
        G.train()
        D.train()
        metric_accum = {
            "generator_loss": 0,
            "discriminator_loss_real": 0,
            "discriminator_loss_fake": 0,
        }
        batch_count = 0
        for i, (X, Y, P, PL) in enumerate(real_dl):
            bs = X.size(0)

            # Samples
            real_samples = X.to(DEVICE)
            latent_samples = self.sample_Z(bs, seq_length, latent_dim).to(DEVICE)
            fake_samples = G(latent_samples)

            # Labels
            real_labels = Y.to(DEVICE)
            fake_labels = torch.full((bs, seq_length, 1), anomaly_label).float().to(DEVICE)
            all_labels = torch.cat([real_labels, fake_labels])

            # Discriminator update
            D_optimizer.zero_grad()
            real_d = D(real_samples)
            fake_d = D(fake_samples.detach())
            all_d = torch.cat([real_d, fake_d])

            d_loss_real = loss_fn(real_d.view(-1), real_labels.view(-1))
            d_loss_fake = loss_fn(fake_d.view(-1), fake_labels.view(-1))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()

            D_optimizer.step()

            # Genertor update
            G_optimizer.zero_grad()
            fake_d = D(fake_samples)
            g_loss = loss_fn(fake_d.view(-1), real_labels.view(-1))
            g_loss.backward()
            G_optimizer.step()

            # Save metrics
            metric_accum['generator_loss'] += g_loss.item()
            metric_accum['discriminator_loss_real'] += d_loss_real.item()
            metric_accum['discriminator_loss_fake'] += d_loss_fake.item()
            batch_count += 1
        D.zero_grad()
        G.zero_grad()
        out = 'G_loss: ' + str(metric_accum['generator_loss'] / batch_count) + ', ' + 'D_loss_real: ' + str(
            metric_accum['discriminator_loss_real'] / batch_count) + ', ' + 'D_loss_fake: ' + str(
            metric_accum['discriminator_loss_fake'] / batch_count)
        print('Epoch ' + str(epoch) + 'training:')
        print(out)

    def train(self, seq_length, latent_dim, train_dl, test_dl, D, G, D_optim, G_optim, loss_fn, random_seed,
              num_epochs, DEVICE, model_dir: Path = Path("models/madgan")) -> None:
        self.set_seed(random_seed)
        model_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            self.train_epoch(G, D, loss_fn, train_dl, G_optim, D_optim, seq_length, latent_dim, DEVICE,
                             normal_label=0, anomaly_label=1, epoch=epoch)
            self.evaluate(G, D, loss_fn, test_dl, seq_length, latent_dim, DEVICE, normal_label=0, anomaly_label=1)

            G.save(model_dir / f"generator_{epoch}.pt")
            D.save(model_dir / f"discriminator_{epoch}.pt")