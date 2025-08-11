import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
import os
import yaml
from tqdm import tqdm
import pandas as pd

from models.vae import Encoder, Decoder, VGGPerceptualLoss, weights_init

def setup_directories(log_dir):
    """Creates necessary directories for logging and saving models."""
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "reconstructions"), exist_ok=True)

def get_dataloader(config):
    """Prepares and returns the CelebA dataloader."""
    data_path = os.path.join(config['DATA_ROOT'], "img_align_celeba")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please place the CelebA dataset in the correct directory.")

    transform = transforms.Compose([
        transforms.Resize(config['IMG_SIZE']),
        transforms.CenterCrop(config['IMG_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(root=config['DATA_ROOT'], transform=transform)
    return DataLoader(dataset=dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=config['NUM_WORKERS'], pin_memory=True)

def save_checkpoint(encoder, decoder, optimizer, epoch, path):
    """Saves a training checkpoint."""
    state = {'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, path)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(encoder, decoder, optimizer, path):
    """Loads a training checkpoint."""
    start_epoch = 0
    if os.path.exists(path):
        print(f"Resuming from checkpoint: {path}")
        checkpoint = torch.load(path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    return start_epoch

def log_to_csv(log_path, metrics):
    """Logs metrics to a CSV file."""
    df = pd.DataFrame([metrics])
    header = not os.path.exists(log_path)
    df.to_csv(log_path, mode='a', header=header, index=False)

def save_validation_images(epoch, fixed_batch, encoder, decoder, log_dir, device):
    """Saves a batch of reconstructed images for validation."""
    with torch.no_grad():
        fixed_batch = fixed_batch.to(device)
        mu, _ = encoder(fixed_batch)
        recon_images = decoder(mu).cpu()
        comparison = torch.cat([fixed_batch[:8].cpu(), recon_images[:8].cpu()])
        save_path = os.path.join(log_dir, "reconstructions", f"reconstruction_{epoch+1}.png")
        save_image(comparison, save_path, nrow=8, normalize=True)

def train(config):
    """Main training loop for the VAE."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    log_dir = config['TRAIN_LOG_DIR']
    setup_directories(log_dir)
    dataloader = get_dataloader(config)

    encoder = Encoder(config['LATENT_DIM']).to(device)
    decoder = Decoder(config['LATENT_DIM']).to(device)
    encoder.apply(weights_init)
    decoder.apply(weights_init)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=config['LEARNING_RATE'], betas=(config['BETA1'], 0.999))

    vgg_loss_fn = VGGPerceptualLoss().to(device)
    mse_loss_fn = nn.MSELoss(reduction='sum')
    kl_div_fn = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    checkpoint_path = os.path.join(log_dir, "checkpoint.pth")
    start_epoch = load_checkpoint(encoder, decoder, optimizer, checkpoint_path)
    fixed_validation_batch, _ = next(iter(dataloader))

    print("Starting VAE Training...")
    for epoch in range(start_epoch, config['NUM_EPOCHS']):
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{config['NUM_EPOCHS']}]")
        total_loss, total_mse, total_vgg, total_kld = 0, 0, 0, 0

        for i, (images, _) in enumerate(loop):
            images = images.to(device)
            bs = images.size(0)
            mu, logvar = encoder(images)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            recon_images = decoder(z)

            mse_loss = mse_loss_fn(recon_images, images) / bs
            vgg_loss = vgg_loss_fn(recon_images, images) * config['VGG_LOSS_WEIGHT'] / bs
            kld_loss = kl_div_fn(mu, logvar) / bs
            loss = mse_loss + vgg_loss + kld_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_vgg += vgg_loss.item()
            total_kld += kld_loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        avg_vgg = total_vgg / len(dataloader)
        avg_kld = total_kld / len(dataloader)
        print(f"End of Epoch {epoch+1} -> Avg Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, VGG: {avg_vgg:.4f}, KLD: {avg_kld:.4f}")

        log_to_csv(os.path.join(log_dir, "training_log.csv"), {'epoch': epoch + 1, 'total_loss': avg_loss, 'mse_loss': avg_mse, 'vgg_loss': avg_vgg, 'kld_loss': avg_kld})
        save_validation_images(epoch, fixed_validation_batch, encoder, decoder, log_dir, device)
        save_checkpoint(encoder, decoder, optimizer, epoch, checkpoint_path)

    print("Training Finished!")

if __name__ == '__main__':
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    train(config)