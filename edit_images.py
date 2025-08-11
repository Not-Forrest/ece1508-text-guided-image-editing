import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import os
import shutil
import yaml
from tqdm import tqdm
import clip
import argparse
from PIL import Image

from models.vae import Encoder, Decoder
from visualize import show_images, show_edit_progression
from train_vae import get_dataloader

def load_editor_models(config, device):
    """Loads the pre-trained VAE encoder and decoder from the specified editing checkpoint."""
    encoder = Encoder(config['LATENT_DIM']).to(device)
    decoder = Decoder(config['LATENT_DIM']).to(device)
    checkpoint_path = config['EDIT_CHECKPOINT_PATH']

    if os.path.exists(checkpoint_path):
        print(f"Loading editor models from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    else:
        raise FileNotFoundError(f"Editor checkpoint not found at '{checkpoint_path}'. Please check the path in config.yaml.")
    encoder.eval()
    decoder.eval()
    return encoder, decoder

def load_image_from_path(path, img_size):
    """Loads and transforms an image from a file path."""
    try:
        image = Image.open(path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return transform(image)
    except FileNotFoundError:
        print(f"Error: Image file not found at {path}")
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def edit_image(initial_image_tensor, target_text, encoder, decoder, device, output_dir, **kwargs):
    """Performs CLIP-guided latent space optimization to edit an image."""
    # Editable parameters with defaults
    steps = kwargs.get('steps', 300)
    lr = kwargs.get('lr', 0.07)
    lambda_reg = kwargs.get('lambda_reg', 0.005)
    lambda_recon = kwargs.get('lambda_recon', 0.1)
    num_augs = kwargs.get('num_augs', 8)
    save_every = kwargs.get('save_every', 25)
    negative_text = kwargs.get('negative_text', None)
    directional = kwargs.get('directional', False)

    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # Setup directories for saving intermediate steps
    run_name = f"edit_{target_text.replace(' ', '_')}"
    intermediate_dir = os.path.join(output_dir, run_name)
    if os.path.exists(intermediate_dir):
        shutil.rmtree(intermediate_dir)
    os.makedirs(intermediate_dir, exist_ok=True)
    print(f"Saving intermediate steps to: {intermediate_dir}")

    # CLIP-specific transformations
    augment_pipe = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
    ])
    clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    # Prepare initial latent code and reconstruction
    x = initial_image_tensor.unsqueeze(0).to(device) if initial_image_tensor.dim() == 3 else initial_image_tensor.to(device)
    with torch.no_grad():
        mu0, _ = encoder(x)
        z0 = mu0.clone().detach()
        initial_recon = decoder(z0).squeeze(0).cpu()
        save_image(initial_recon * 0.5 + 0.5, os.path.join(intermediate_dir, "step_0.png"))

    # Setup optimization
    delta = torch.zeros_like(z0, requires_grad=True, device=device)
    optimizer = optim.Adam([delta], lr=lr)

    # Prepare text embeddings
    with torch.no_grad():
        text_embed = F.normalize(clip_model.encode_text(clip.tokenize([target_text]).to(device)), dim=-1)
        neg_text_embed = None
        if negative_text:
            neg_text_embed = F.normalize(clip_model.encode_text(clip.tokenize([negative_text]).to(device)), dim=-1)
        emb0 = F.normalize(clip_model.encode_image(clip_normalize(F.interpolate((decoder(z0) + 1) / 2, (224, 224), mode='bilinear'))), dim=-1) if directional else None

    # Optimization loop
    loop = tqdm(range(1, steps + 1), desc=f"Editing: '{target_text}'")
    for step in loop:
        z = z0 + delta
        recon = decoder(z)
        views = torch.stack([clip_normalize(augment_pipe(F.interpolate((recon + 1) / 2, (224, 224), mode='bilinear').squeeze(0))) for _ in range(num_augs)])
        img_feat = F.normalize(clip_model.encode_image(views), dim=-1)

        # Calculate losses
        sim_t = (img_feat @ text_embed.T).mean()
        clip_loss = 1 - (sim_t - (img_feat @ emb0.T).mean()) if directional else 1 - sim_t
        if neg_text_embed is not None:
            neg_clip_loss = (img_feat @ neg_text_embed.T).mean()
        reg_loss = delta.pow(2).mean()
        recon_loss = F.mse_loss(recon, x)
        loss = clip_loss + neg_clip_loss + lambda_reg * reg_loss + lambda_recon * recon_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item(), clip_loss=clip_loss.item())
        if step % save_every == 0:
            save_image((decoder(z0 + delta).squeeze(0).cpu() * 0.5 + 0.5), os.path.join(intermediate_dir, f"step_{step}.png"))

    final_image = decoder(z0 + delta).squeeze(0).cpu()
    save_image(final_image * 0.5 + 0.5, os.path.join(intermediate_dir, "final.png"))
    return final_image, intermediate_dir

def main():
    """Main function to parse arguments and run the image editing process."""
    parser = argparse.ArgumentParser(description="Perform text-guided image editing using a VAE and CLIP.")
    
    # Mutually exclusive group for the image source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--index", type=int, help="Index of the image from the CelebA dataset to edit.")
    source_group.add_argument("--image_path", type=str, help="Path to a custom 64x64 image to edit.")

    # Required prompt argument
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing the desired edit.")
    
    # Optional negative prompt
    parser.add_argument("--negative_prompt", type=str, default=None, help="Optional text prompt for attributes to avoid.")
    
    args = parser.parse_args()

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder = load_editor_models(config, device)

    # Load the initial image based on the provided arguments
    if args.index is not None:
        print(f"Loading image at index {args.index} from CelebA dataset...")
        dataloader = get_dataloader(config)
        if args.index >= len(dataloader.dataset):
            print(f"Error: Index {args.index} is out of bounds for the dataset (size: {len(dataloader.dataset)}).")
            return
        initial_image, _ = dataloader.dataset[args.index]
        original_text = f"Original (Index: {args.index})"
    else:  # args.image_path is not None
        print(f"Loading image from path: {args.image_path}")
        initial_image = load_image_from_path(args.image_path, config['IMG_SIZE'])
        if initial_image is None:
            return
        original_text = f"Original ({os.path.basename(args.image_path)})"

    # Perform the edit with the specified prompts
    edited_image, output_dir = edit_image(
        initial_image_tensor=initial_image,
        target_text=args.prompt,
        encoder=encoder,
        decoder=decoder,
        device=device,
        output_dir=config['EDIT_OUTPUT_DIR'],
        negative_text=args.negative_prompt,
        # Make these arguments as well?
        steps=150,
        lr=0.07,
        lambda_recon=0.25,
        save_every=50,
    )

    print(f"\nEditing complete! Final image saved in '{output_dir}'.")
    show_images(initial_image, edited_image, original_text=original_text, edited_text=f"Edited: '{args.prompt}'")
    show_edit_progression(output_dir, title=f"Progression for '{args.prompt}'")

if __name__ == '__main__':
    main()
