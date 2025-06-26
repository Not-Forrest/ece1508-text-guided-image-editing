import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

from models.vae import SimpleConvVAE

class ConditionalResize:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        if img.size != (self.target_size[1], self.target_size[0]):
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

def visualize_reconstructions(config, checkpoint_path, num_samples=8):
    model_name = config["model_name"]
    dataset_name = config["dataset"]
    resize = config["resize"]
    channels = 3 if config.get("color", True) else 1

    transform = transforms.Compose([
        ConditionalResize(resize),
        ConditionalGrayscale(enabled=(channels == 1)),
        transforms.ToTensor()
    ])

    if dataset_name.lower() == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    elif dataset_name.lower() == "cifar100":
        dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)
    images, _ = next(iter(dataloader))

    image_size = (channels, resize[0], resize[1])
    if model_name == "simple_conv_vae":
        model = SimpleConvVAE(input_shape=image_size)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        images = images.to(device)
        recon, _, _ = model(images)

    # Convert to CPU for plotting
    images = images.cpu()
    recon = recon.cpu()

    fig, axs = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    for i in range(num_samples):
        axs[0, i].imshow(images[i].permute(1, 2, 0).squeeze(), cmap="gray" if channels == 1 else None)
        axs[0, i].axis("off")
        axs[0, i].set_title("Original")

        axs[1, i].imshow(recon[i].permute(1, 2, 0).squeeze(), cmap="gray" if channels == 1 else None)
        axs[1, i].axis("off")
        axs[1, i].set_title("Reconstructed")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to visualize")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    visualize_reconstructions(config, args.checkpoint, args.num_samples)
