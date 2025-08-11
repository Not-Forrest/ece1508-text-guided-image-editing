import torch
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channel = 32, embedding_dim = 256):
        super().__init__()
        self.downconv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channel, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
        )

        self.downconv2 = nn.Sequential(
            nn.Conv2d(hidden_channel, hidden_channel*4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
        )
        self.downconv3 = nn.Sequential(
            nn.Conv2d(hidden_channel*4, embedding_dim, 3, padding=1), 
        )

        self.param_init()
    def param_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):

        x = self.downconv1(x)
        x = self.downconv2(x) 
        x = self.downconv3(x) 

        return x

class Decoder(nn.Module):
    def __init__(self, embedding_dim=256, hidden_channels=32, out_channels=3):
        super().__init__()

        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_channels*4, kernel_size=3, padding=1),  
            nn.ReLU()
        )
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels *4, hidden_channels, kernel_size=4, stride=2, padding=1),    
            nn.ReLU()
        )
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels , out_channels, kernel_size=4, stride=2, padding=1), 
            nn.ReLU()
        )
        self.finalconv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  
        )
        self.param_init()
    def param_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):

        x = self.upconv3(x) 
        x = self.upconv2(x)  
        x = self.upconv1(x)  
        x = self.finalconv(x) 

        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_loss_weight):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_loss_weight = commitment_loss_weight

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.param_init()  # call initialization

    def param_init(self):
        # He initialization for embedding weights
        nn.init.kaiming_normal_(
            self.codebook.weight,
            mode='fan_in',
            nonlinearity='relu'
        )

    def forward(self, u):
        B, C, H, W = u.shape
        u_perm = u.permute(0, 2, 3, 1).contiguous()
        flat_u = u_perm.view(-1, C)

        distances = torch.cdist(flat_u, self.codebook.weight)
        encoding_indices = torch.argmin(distances, dim=1)
        z_quantized = self.codebook(encoding_indices)

        z_quantized = z_quantized.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        z_train = z_quantized + (u - u.detach())

        codebook_loss = F.mse_loss(z_quantized, u.detach())
        commitment_loss = F.mse_loss(u, z_quantized.detach())
        vq_loss = codebook_loss + self.commitment_loss_weight * commitment_loss

        return {
            'u': u,
            'z': z_train,
            'vq_loss': vq_loss,
            'encoding_indices': encoding_indices.view(B, H, W)
        }