import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvVAE(nn.Module):
    def __init__(self, latent_dim=128, input_shape=(3, 32, 32)):
        super().__init__()
        self.latent_dim = latent_dim
        c, h, w = input_shape
        self.c, self.h, self.w = c, h, w

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, 4, 2, 1),  # -> 32 x H/2 x W/2
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # -> 64 x H/4 x W/4
            nn.ReLU(),
        )

        # Infer flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            enc_out = self.encoder(dummy)
            self.enc_shape = enc_out.shape[1:]  # (C, H', W')
            self.flattened_size = enc_out.numel()

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # → 32 × H/2 × W/2
            nn.ReLU(),
            nn.ConvTranspose2d(32, c, 4, 2, 1),   # → C × H × W
            nn.Sigmoid()  # Make sure output is in [0,1]
        )

    def encode(self, x):
        h = self.encoder(x)
        h_flat = self.flatten(h)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, *self.enc_shape)  # unflatten
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar