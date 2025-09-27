# 🧠 Synthetic MRI Generator (Pix2Pix GAN)

This is my implementation of a **Pix2Pix Conditional GAN** for image-to-image translation, specifically focusing on synthesizing **T1ce-weighted MRI scans** from **T2-weighted MRI scans**.

The goal is to generate realistic T1ce images given T2 inputs, using a U-Net Generator and a PatchGAN Discriminator.

-----

## 🛠️ Setup and Installation

I manage all project dependencies using the provided `environment.yml` file.

1.  **Create the Environment:**
    ```bash
    conda env create -f environment.yml
    ```
2.  **Activate the Environment:**
    ```bash
    conda activate synthetic_mri_gan
    ```
3.  **Verify Setup (Optional):**
    I can run `jupyter notebook` and use the `01_data_exploration.ipynb` file to verify the data loading and preprocessing steps.

-----

## 💾 Data Flow

My pipeline expects raw NIfTI data and converts it into optimized NumPy slices for TensorFlow.

1.  **Place Data:** I place my raw MRI subject folders (containing T2 and T1ce files) into the `data/raw/` directory.
2.  **Preprocess:** I run the dedicated script to process, normalize, and split the data.
    ```bash
    python src/preprocess.py
    ```
    *Result:* Processed slices are saved to `data/processed/`.

-----

## 🏃 Training and Resumption

I use the `run_train.sh` script to manage my training pipeline.

| Command | Purpose |
| :--- | :--- |
| `bash run_train.sh` | Starts preprocessing, then launches training (`src/train.py`). |
| **Resumption** | **If I stop training,** simply rerunning `bash run_train.sh` automatically loads the latest checkpoint (`ckpt-N`) and resumes training from the correct epoch. |
| **Results** | Generated sample images are saved to `results/samples/`. |

-----

## ✅ Evaluation

To assess the quality of the final synthesized T1ce images, I run the dedicated evaluation script.

```bash
python src/evaluate.py
```

This script loads the best saved model and calculates standard image quality metrics (PSNR and SSIM) on the held-out test set.