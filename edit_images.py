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

from models.vae import Encoder, Decoder
from visualize import show_images, show_edit_progression
from train_vae import get_dataloader

def load_trained_vae(config, device):
    """Loads the pre-trained VAE encoder and decoder from a checkpoint."""
    encoder = Encoder(config['LATENT_DIM']).to(device)
    decoder = Decoder(config['LATENT_DIM']).to(device)
    checkpoint_path = os.path.join(config['LOG_DIR'], "checkpoint.pth")

    if os.path.exists(checkpoint_path):
        print(f"Loading pre-trained VAE from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please train the VAE first by running train_vae.py")
    encoder.eval()
    decoder.eval()
    return encoder, decoder

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
    text_embed = F.normalize(clip_model.encode_text(clip.tokenize([target_text]).to(device)), dim=-1)
    neg_text_embed = F.normalize(clip_model.encode_text(clip.tokenize([negative_text]).to(device)), dim=-1) if negative_text else None
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
        neg_clip_loss = (img_feat @ neg_text_embed.T).mean() if neg_text_embed is not None else 0
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

if __name__ == '__main__':
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder = load_trained_vae(config, device)
    dataloader = get_dataloader(config)

    # --- Perform an example edit ---
    image_index = 150 # Which image from the dataset to edit
    initial_image, _ = dataloader.dataset[image_index]
    target_text = "a photo of a smiling woman with blonde hair"

    edited_image, output_dir = edit_image(
        initial_image_tensor=initial_image,
        target_text=target_text,
        encoder=encoder,
        decoder=decoder,
        device=device,
        output_dir=config['EDIT_OUTPUT_DIR'],
        steps=500,
        lr=0.07,
        lambda_recon=0.25,
        save_every=50,
    )

    print(f"\nEditing complete! Final image saved in '{output_dir}'.")
    show_images(initial_image, edited_image, original_text=f"Original (Index: {image_index})", edited_text=f"Edited: '{target_text}'")
    show_edit_progression(output_dir, title=f"Progression for '{target_text}'")