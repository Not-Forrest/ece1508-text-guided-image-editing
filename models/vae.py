import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvVAE(nn.Module):
    def __init__(self, latent_dim=128, input_shape=(3, 32, 32)):
        super().__init__()
        self.latent_dim = latent_dim
        c, h, w = input_shape
        self.h, self.w = h, w
        self.flattened_size = 64 * (h // 4) * (w // 4)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, 4, 2, 1),  # -> 32 x H/2 x W/2
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # -> 64 x H/4 x W/4
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # -> 32 x H/2 x W/2
            nn.ReLU(),
            nn.ConvTranspose2d(32, c, 4, 2, 1),   # -> C x H x W
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        size_h = self.h // 4
        size_w = self.w // 4
        h = h.view(-1, 64, size_h, size_w)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar