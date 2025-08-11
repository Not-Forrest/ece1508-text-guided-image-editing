from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch.optim.lr_scheduler import StepLR
import math
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision.datasets import CelebA
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
from models.vq_vae import Encoder, Decoder, VectorQuantizer
from models.mapper import CAMapper, ConvMapper, CLIPTextEncoder
import random
from latent_mapper_dataset import train_loader, val_loader, test_loader




def train_VQVAE(encoder, decoder, vq_layer, mapper, text_encoder, num_epochs, train_loader, optimizer,scheduler = None, device=DEVICE,
                recon_weight=1.0, vq_weight=0.25, mapper_weight = 1.0):

    encoder.to(device)
    decoder.to(device)
    vq_layer.to(device)
    mapper.to(device)

    encoder.train()
    decoder.train()
    vq_layer.train()
    mapper.train()

    text_encoder.to(device).eval()

    # training loop
    for epoch in range(num_epochs):
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for _, batch in enumerate(pbar):
            src_img = batch["image"].to(device)
            texts = batch["text"]
            tgt_img = batch["target"].to(device)



            with torch.no_grad():
                z_txt = text_encoder(texts)

            u = encoder(src_img)
            u_pred = mapper(u, z_txt)
            vq_output = vq_layer(u_pred)
            z_q = vq_output['z']

            recon_images = decoder(z_q)


            recon_loss = F.mse_loss(recon_images, tgt_img)



            vq_loss = vq_output['vq_loss']

            loss = recon_weight * recon_loss + vq_weight * vq_loss

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            if epoch > 5:
                if scheduler is not None:
                    scheduler.step()

            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_loss += loss.item()

            pbar.set_postfix({
                'recon_loss': f"{recon_loss.item():.4f}",
                'vq_loss': f"{vq_loss.item():.4f}",
                'total_loss': f"{loss.item():.4f}"
            })


        avg_recon = total_recon_loss / len(train_loader)
        avg_vq = total_vq_loss / len(train_loader)
        avg_total = total_loss / len(train_loader)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Recon Loss: {avg_recon:.5f}")
        print(f"  Avg VQ Loss: {avg_vq:.5f}")
        print(f"  Avg Total Loss: {avg_total:.5f}")

        # save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'vq_layer': vq_layer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_total,
                'mapper': mapper.state_dict(),
            }
            torch.save(checkpoint, f"vqvae_checkpoint_epoch_{epoch+1}.pth")

    print("Training completed!")
def train_mapper(encoder, decoder, vq_layer, mapper, text_encoder, num_epochs, train_loader, optimizer, device=DEVICE):

    encoder.to(device)
    decoder.to(device)
    vq_layer.to(device)
    mapper.to(device)

    encoder.eval()
    decoder.eval()
    vq_layer.eval()
    mapper.train()

    text_encoder.to(device).eval()

    # training loop
    for epoch in range(num_epochs):
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for _, batch in enumerate(pbar):
            src_img = batch["image"].to(device)
            texts = batch["text"]
            tgt_img = batch["target"].to(device)



            with torch.no_grad():
                z_txt = text_encoder(texts)

            u = encoder(src_img)
            u_tgt = encoder(tgt_img)
            u_pred = mapper(u, z_txt)

            # total loss
            loss = F.mse_loss(u_pred, u_tgt)

            optimizer.zero_grad()
            loss.backward()


            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix({
                'mapper_loss': f"{loss.item():.4f}"
            })


        avg_total = total_loss / len(train_loader)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Total Loss: {avg_total:.5f}")

        # save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'vq_layer': vq_layer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_total,
                'mapper': mapper.state_dict(),
            }
            torch.save(checkpoint, f"vqvae_checkpoint_epoch_{epoch+1}.pth")

    print("Training completed!")
if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_mapper = CAMapper()
    model_decoder = Decoder()
    model_encoder = Encoder()
    model_text_encoder = CLIPTextEncoder()
    vq_layer = VectorQuantizer(num_embeddings=128, embedding_dim=256, commitment_loss_weight=0.25)
    params = list(model_encoder.parameters()) + list(model_decoder.parameters()) + list(vq_layer.parameters()) + list(model_mapper.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    mapper_optimizer = torch.optim.AdamW(model_mapper.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    train_VQVAE(encoder=model_encoder, decoder=model_decoder, vq_layer=vq_layer,mapper = model_mapper, text_encoder=model_text_encoder, num_epochs=10, train_loader=train_loader, optimizer=optimizer)
    # train_mapper(encoder=model_encoder, decoder=model_decoder, vq_layer=vq_layer,mapper = model_mapper, text_encoder=model_text_encoder, num_epochs=5, train_loader=train_loader, optimizer=mapper_optimizer)
    
