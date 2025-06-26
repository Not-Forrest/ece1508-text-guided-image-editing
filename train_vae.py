import os
import csv
import yaml
import signal
import argparse
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from models.vae import SimpleConvVAE

# Early stopping signal
early_stop = False
def signal_handler(sig, frame):
    global early_stop
    early_stop = True
    print("Early stopping triggered. Will finish this epoch.")

signal.signal(signal.SIGINT, signal_handler)

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")

def log_metrics(csv_file, metrics):
    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

# TODO Move this into data loader file
def load_dataset(name, transform, batch_size):
    if name.lower() == "cifar10":
        train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        val_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif name.lower() == "cifar100":
        train_data = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        val_data = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    return train_loader, val_loader

# TODO - move this into data transform / loader file
class ConditionalResize:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        if img.size != (self.target_size[1], self.target_size[0]):  # PIL uses (W, H)
            return F.resize(img, self.target_size)
        return img

class ConditionalGrayscale:
    def __init__(self, enabled, output_channels=1):
        self.enabled = enabled
        self.output_channels = output_channels

    def __call__(self, img):
        if not self.enabled:
            return img
        if img.mode != 'L' and self.output_channels == 1:
            return F.to_grayscale(img, num_output_channels=self.output_channels)
        return img

def train_vae(config_path, checkpoint_path=None):
    config = load_config(config_path)

    model_name = config["model_name"]
    dataset_name = config["dataset"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    lr = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    checkpoint_interval = config["checkpoint_interval"]
    resize = config["resize"]
    channels = 3 if config.get("color", True) else 1

    # Output directory: out/<timestamp>_<model_name>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("out", f"{timestamp}_{model_name}")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics.csv")

    # TODO - this should be part of the data loader!
    # Dataset
    transform = transforms.Compose([
        ConditionalResize(resize),
        ConditionalGrayscale(enabled=(channels == 1)),
        transforms.ToTensor()
    ])
    train_loader, val_loader = load_dataset(dataset_name, transform, batch_size)
    image_size = (channels, resize[0], resize[1])

    # TODO - this should be taken care of in another function
    # Model selection
    if model_name == "simple_conv_vae":
        model = SimpleConvVAE(input_shape=image_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Loaded checkpoint from {checkpoint_path}")

    for epoch in range(start_epoch, num_epochs):
        if early_stop:
            break

        # Training
        model.train()
        train_recon, train_kl, train_total = 0, 0, 0
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            x = x.to(device)
            x_recon, mu, logvar = model(x)

            recon_loss = criterion(x_recon, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            total_loss = recon_loss + kl_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
            train_total += total_loss.item()

        train_recon /= len(train_loader)
        train_kl /= len(train_loader)
        train_total /= len(train_loader)

        # Validation
        model.eval()
        val_recon, val_kl, val_total = 0, 0, 0
        with torch.no_grad():
            for x, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                x = x.to(device)
                x_recon, mu, logvar = model(x)

                recon_loss = criterion(x_recon, x)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
                total_loss = recon_loss + kl_loss

                val_recon += recon_loss.item()
                val_kl += kl_loss.item()
                val_total += total_loss.item()

        val_recon /= len(val_loader)
        val_kl /= len(val_loader)
        val_total /= len(val_loader)

        print(f"Epoch {epoch+1} Summary: Train Loss = {train_total:.4f}, Val Loss = {val_total:.4f}")

        metrics = {
            "epoch": epoch + 1,
            "train_recon_loss": train_recon,
            "train_kl_loss": train_kl,
            "train_total_loss": train_total,
            "val_recon_loss": val_recon,
            "val_kl_loss": val_kl,
            "val_total_loss": val_total
        }
        log_metrics(csv_path, metrics)

        # Checkpoint
        if (epoch + 1) % checkpoint_interval == 0 or early_stop:
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, os.path.join(output_dir, f"{model_name}_epoch_{epoch+1}.pt"))

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vae_default.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to optional checkpoint file")
    args = parser.parse_args()

    train_vae(args.config, args.checkpoint)
