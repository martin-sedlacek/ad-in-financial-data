import numpy as np
import torch
import random
from typing import Dict
from pathlib import Path
from models.MADGAN import AnomalyDetector
from utils.evaluation import excess_mass, mass_volume


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
        for i, (X, Y) in enumerate(real_dl):
            bs = X.size(0)

            # Samples
            real_samples = X.to(DEVICE)
            latent_samples = self.sample_Z(bs, seq_length, latent_dim).to(DEVICE)
            fake_samples = G(latent_samples)

            # Labels
            real_labels = torch.full((bs, seq_length, 1), normal_label).float().to(DEVICE)
            fake_labels = torch.full((bs, seq_length, 1), anomaly_label).float().to(DEVICE)

            # Discriminator update
            D_optimizer.zero_grad()
            real_d = D(real_samples)
            fake_d = D(fake_samples.detach())

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
        print('Epoch ' + str(epoch) + ' training:')
        print(out)

    def train(self, seq_length, latent_dim, tscv_dl_list, D, G, D_optim, G_optim, loss_fn, random_seed,
              num_epochs, DEVICE, model_dir: Path = Path("models/madgan")) -> None:
        self.set_seed(random_seed)

        ad = AnomalyDetector(discriminator=D, generator=G, latent_space_dim=latent_dim, anomaly_threshold=0.5, DEVICE=DEVICE)

        for train_dl, test_dl in tscv_dl_list:
            for epoch in range(num_epochs):
                self.train_epoch(G, D, loss_fn, train_dl, G_optim, D_optim, seq_length, latent_dim, DEVICE,
                                 normal_label=0, anomaly_label=1, epoch=epoch)
            self.evaluate(G, D, loss_fn, test_dl, seq_length, latent_dim, DEVICE, normal_label=0, anomaly_label=1)
            total_em = total_mv = 0
            for X, Y in test_dl:
                scores = self.MAD_GAN_EMMV(ad, X.to(DEVICE), DEVICE=DEVICE)
                total_em += scores['em']
                total_mv += scores['mv']
            print("EMMV evaluation:")
            print(total_em / len(test_dl), total_mv / len(test_dl))

    def MAD_GAN_EMMV(self, trained_model, x, n_generated=10000, alpha_min=0.9, alpha_max=0.999, t_max=0.9, DEVICE="cpu"):
        # Get limits and volume support.
        lim_inf = torch.min(x.view(-1, x.size(-1)), dim=0)[0]
        lim_sup = torch.max(x.view(-1, x.size(-1)), dim=0)[0]
        offset = 1e-60  # to prevent division by 0

        # Volume support
        volume_support = torch.prod(lim_sup - lim_inf).item() + offset

        # Determine EM and MV parameters
        t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
        axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)

        # Get anomaly scores
        anomaly_score = trained_model.predict(x).view(-1, 1).detach().cpu().numpy()

        reducer = 10
        reduced_n = int(n_generated / reducer)
        s_unif_list = []
        for i in range(reducer):
            unif = torch.rand(reduced_n, x.size(1), x.size(2)).to(DEVICE)
            m = lim_sup - lim_inf
            unif = unif * m
            unif = unif + lim_inf
            s_unif = trained_model.predict(unif).view(-1, 1).detach()
            s_unif_list.append(s_unif)
        s_unif_total = torch.cat(s_unif_list).cpu().numpy()

        # Get EM and MV scores
        AUC_em, em, amax = excess_mass(t, t_max, volume_support, s_unif_total, anomaly_score, n_generated)
        AUC_mv, mv = mass_volume(axis_alpha, volume_support, s_unif_total, anomaly_score, n_generated)

        # Return a dataframe containing EMMV information
        scores = {
            'em': np.mean(em),
            'mv': np.mean(mv),
        }
        return scores

    @torch.no_grad()
    def evaluate(self, G, D, loss_fn, real_dl, seq_length, latent_dim, DEVICE, normal_label: int = 0, anomaly_label: int = 1) -> Dict[str, float]:
        metric_accum = {
            "D_loss": 0,
            "G_acc": 0,
            "D_acc": 0
        }
        batch_count = 0
        for X, Y in real_dl:
            bs = X.size(0)

            # Samples
            real_samples = X.to(DEVICE)
            latent_samples = self.sample_Z(bs, seq_length, latent_dim).to(DEVICE)
            fake_samples = G(latent_samples)

            # Labels
            real_labels = torch.full((bs, seq_length, 1), normal_label).float().to(DEVICE)
            fake_labels = torch.full((bs, seq_length, 1), anomaly_label).float().to(DEVICE)
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
            discriminator_acc = discriminator_acc.sum().div(2 * bs * seq_length)

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
