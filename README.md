# Text-Guided Image Editing with VAEs and CLIP

This project implements a text-guided image editing pipeline using a Variational Autoencoder (VAE) trained on the CelebA dataset and guided by OpenAI's CLIP model.

---

## 1. Installation

First, install all the necessary Python libraries using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

---

## 2. Dataset Setup

### Step 1: Download the Dataset
Download the "Align&Cropped Images" version of the CelebA dataset from the official source.
* **Download Link:** [CelebA Dataset on Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ)
* Download the `img_align_celeba.zip` file.

### Step 2: Place the Dataset Correctly
1.  Unzip the `img_align_celeba.zip` file. This will give you a folder named `img_align_celeba`.
2.  Place this folder into the correct directory within the project so that the final path is `data/celeba/img_align_celeba/`. You may need to create the `data` and `celeba` folders first.

---

## 3. How to Use

You can either train a new VAE model or use the provided pre-trained model to start editing images immediately.

### Option A: Train a New VAE

To train the Variational Autoencoder from scratch on the CelebA dataset, run the following command:
```bash
python train_vae.py
```
* Training progress, logs (`training_log.csv`), and model checkpoints (`checkpoint.pth`) will be saved to the directory specified by `TRAIN_LOG_DIR` in `config.yaml` (default is `outputs/logs/run_1/`).

### Option B: Edit Images with a Pre-trained Model

To edit an image using a text prompt, run the `edit_images.py` script. This script will load the pre-trained model by default and save all outputs to the `outputs/edits/` directory.

#### Examples:

* **Edit a dataset image by index:**
    ```bash
    python edit_images.py --prompt "blonde woman" --negative_prompt "hat" --index 150
    ```

* **Edit a local image file (must be 64x64):**
    ```bash
    python edit_images.py --prompt "a person with a beard" --image_path "./my_images/face.png" --negative_prompt "a photo of a woman"
    ````