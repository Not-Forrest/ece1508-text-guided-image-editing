import torch
import random
from edit_images import val_loader, test_loader
from tqdm import tqdm
import matplotlib.pyplot as plt

@torch.no_grad()
def sample_edit(encoder, decoder, vq_layer, mapper, text_encoder, img_src, text_prompts, device):
    """
    Apply text-guided edit to a source image using the trained mapper.

    Args:
        mapper: trained mapper model
        encoder: frozen encoder
        decoder: frozen decoder
        text_encoder: CLIPTextEncoder
        img_src: tensor (B, 3, H, W)
        text_prompt: list of strings, length B
        device: cuda or cpu

    Returns:
        img_pred: edited image (B, 3, H, W)
    """
    text_encoder.eval().to(device)
    encoder.eval().to(device)
    decoder.eval().to(device)
    mapper.eval().to(device)
    vq_layer.eval().to(device)

    # Move to device
    img_src = img_src.to(device)

    # Encode image
    with torch.no_grad():
        z_txt = text_encoder(text_prompts)

        u = encoder(img_src)
        u_pred = mapper(u, z_txt)
        # print("get here", u.shape)

        vq_output = vq_layer(u_pred)
        z_q = vq_output['z']

        recon_images = decoder(z_q)

    return recon_images


checkpoint_path = f"vqvae_checkpoint_epoch_20.pth"
vq_model = torch.load(checkpoint_path, map_location='cuda')
model_encoder = vq_model['encoder']
model_decoder = vq_model['decoder']
model_mapper = vq_model['mapper']
model_text_encoder = vq_model['text_encoder']
vq_layer = vq_model['vq_layer']

def show_comparison(image_src, pred_img, image_tgt, captions, max_samples=4):
    """
    横向排列展示 Source → Prompt → Generated → Target。
    """
    num_samples = min(len(image_src), max_samples)
    plt.figure(figsize=(16, 4 * num_samples))

    for i in range(num_samples):
        # Source
        plt.subplot(num_samples, 4, i * 4 + 1)
        plt.imshow(image_src[i].permute(1, 2, 0).cpu().numpy())
        plt.title("Source")
        plt.axis("off")

        # Prompt
        plt.subplot(num_samples, 4, i * 4 + 2)
        plt.text(0.5, 0.5, captions[i], wrap=True, ha="center", va="center", fontsize=10)
        plt.title("Prompt")
        plt.axis("off")

        # Generated
        plt.subplot(num_samples, 4, i * 4 + 3)
        plt.imshow(pred_img[i].permute(1, 2, 0).detach().cpu().numpy())
        plt.title("Generated")
        plt.axis("off")

        # Target
        plt.subplot(num_samples, 4, i * 4 + 4)
        plt.imshow(image_tgt[i].permute(1, 2, 0).cpu().numpy())
        plt.title("Target")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
def test(model_decoder, model_encoder, model_mapper, vq_layer, text_encoder, dataloader, device=DEVICE):
    model_encoder.eval()
    model_decoder.eval()
    text_encoder.eval()
    model_mapper.eval()
    model_encoder.to(device)
    model_decoder.to(device)
    text_encoder.to(device)
    model_mapper.to(device)
    pred_imgs = []
    target_imgs = []
    losses = []
    printing_img = True
    with torch.no_grad():
        for batch in dataloader:
            image_src = batch["image"].to(device)
            image_tgt = batch["target"].to(device)
            text = batch["text"]

            img_pred = sample_edit(model_encoder, model_decoder, vq_layer, model_mapper, text_encoder, image_src, text, device)
            loss = F.mse_loss(img_pred, image_tgt)
            losses.append(loss.item())
            pred_imgs.append(img_pred.cpu())
            target_imgs.append(image_tgt.cpu())

            if printing_img:
                show_comparison(image_src, img_pred, image_tgt, text)
                printing_img = False
                break