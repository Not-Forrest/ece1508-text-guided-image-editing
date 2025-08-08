import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

def weights_init(m):
    """Custom weight initialization."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
    def forward(self, x):
        x = self.main(x); x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Tanh()
        )
    def forward(self, x):
        x = self.fc(x); x = x.view(x.size(0), 512, 4, 4)
        return self.main(x)

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features.eval()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for param in self.parameters():
            param.requires_grad = False
        self.resize = resize
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, x, y):
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        h_x = self.slice1(x)
        h_y = self.slice1(y)
        loss1 = self.loss(h_x, h_y)
        h_x = self.slice2(h_x)
        h_y = self.slice2(h_y)
        loss2 = self.loss(h_x, h_y)
        h_x = self.slice3(h_x)
        h_y = self.slice3(h_y)
        loss3 = self.loss(h_x, h_y)
        return loss1 + loss2 + loss3