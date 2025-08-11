import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image

def show_images(original_img, edited_img, original_text="Original", edited_text="Edited"):
    if original_img.dim() > 3: original_img = original_img.squeeze(0)
    if edited_img.dim() > 3: edited_img = edited_img.squeeze(0)
    original_img = (original_img + 1) / 2
    edited_img = (edited_img + 1) / 2
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_img.permute(1, 2, 0).cpu().detach().numpy())
    ax[0].set_title(original_text)
    ax[0].axis('off')
    ax[1].imshow(edited_img.permute(1, 2, 0).cpu().detach().numpy())
    ax[1].set_title(edited_text)
    ax[1].axis('off')
    plt.show()

def show_edit_progression(output_dir, title="Edit Progression", grid_columns=8):
    image_files = [f for f in os.listdir(output_dir) if f.startswith('step_') and f.endswith('.png')]
    sorted_filenames = sorted(image_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    if not sorted_filenames:
        print("No images found in the output directory.")
        return
    full_paths = [os.path.join(output_dir, f) for f in sorted_filenames]
    images = [transforms.ToTensor()(Image.open(f)) for f in full_paths]
    grid = make_grid(images, nrow=grid_columns, normalize=True)
    plt.figure(figsize=(18, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.show()