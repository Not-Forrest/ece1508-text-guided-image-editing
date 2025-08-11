import torch
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, random_split
import random

EDITABLE_ATTRS = ["Smiling", "Eyeglasses", "Bangs", "Blond_Hair", "Male"]

transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize([0.5]*3, [0.5]*3)
])

celeba = CelebA(
    root='./data',
    split='train',
    target_type='attr',
    transform=transform,
    download=True
)

attr_names = celeba.attr_names
edit_attr_indices = [attr_names.index(attr) for attr in EDITABLE_ATTRS]

# Maps from attribute tuple to list of image indices
attr_index = defaultdict(list)

for idx in range(len(celeba)):
    attr_vec = celeba.attr[idx][edit_attr_indices].numpy().astype(int)
    key = tuple(attr_vec)
    attr_index[key].append(idx)

class CelebAEditDataset(Dataset):
    def __init__(self, celeba, attr_index, edit_attr_indices, transform=None):
        self.celeba = celeba
        self.attr_index = attr_index
        self.edit_attr_indices = edit_attr_indices
        self.attr_names = [celeba.attr_names[i] for i in edit_attr_indices]
        self.transform = transform

    def __len__(self):
        return len(self.celeba)

    def __getitem__(self, idx):
        img_src, attr_src = self.celeba[idx]
        attr_src = attr_src[self.edit_attr_indices].numpy().astype(int)
        attr_src_key = tuple(attr_src)

        # Randomly choose one attribute to flip
        edit_attr = random.randint(0, len(attr_src) - 1)
        attr_tgt = attr_src.copy()
        attr_tgt[edit_attr] = 1 - attr_tgt[edit_attr]  # flip the bit

        attr_tgt_key = tuple(attr_tgt)

        # Find a matching target image with new attributes
        candidates = self.attr_index.get(attr_tgt_key, [])
        if not candidates:
            return self.__getitem__((idx + 1) % len(self))  # fallback

        tgt_idx = random.choice(candidates)
        img_tgt, _ = self.celeba[tgt_idx]

        # Text prompt
        attr_name = self.attr_names[edit_attr].replace("_", " ").lower()
        prompt = f"add {attr_name}" if attr_tgt[edit_attr] == 1 else f"remove {attr_name}"

        return {"image": img_src, "text": prompt, "target": img_tgt}

if __name__ == '__main__':
    celeba_dataset = CelebAEditDataset(
        celeba=celeba,
        attr_index=attr_index,
        edit_attr_indices=edit_attr_indices,
        transform=transform
    )

    total_size = len(celeba_dataset)

    # Calculate sizes for train, val, and test sets
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size  # Ensures total sum is correct

    # Perform random split
    train_dataset, val_dataset, test_dataset = random_split(
        celeba_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
