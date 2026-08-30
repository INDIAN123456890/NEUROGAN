# 🧠 NEUROGAN

### MRI-to-MRI Image Translation using Pix2Pix Conditional GAN

**NEUROGAN** is a deep learning project that generates **T1ce-weighted MRI images from T2-weighted MRI images** using a **Pix2Pix Conditional GAN**.

The project uses a **U-Net Generator** and **PatchGAN Discriminator** to learn the mapping between paired MRI modalities. It includes data preprocessing, GAN training, checkpointing, sample generation, and quantitative evaluation using **PSNR and SSIM**.

> ⚠️ **Disclaimer:** This project is intended for educational and research purposes only. Generated MRI images must not be used for clinical diagnosis or medical decision-making.

---

## ✨ Features

* 🧠 Pix2Pix Conditional GAN for MRI-to-MRI translation
* 🔄 T2 MRI → Synthetic T1ce MRI generation
* 🏗️ U-Net based Generator
* 🔍 PatchGAN Discriminator
* 🧹 NIfTI MRI preprocessing and slice extraction
* 💾 Model checkpointing and training resumption
* 🖼️ Generated image samples
* 📊 PSNR and SSIM evaluation
* ⚡ GPU availability checking
* 🚀 Automated preprocessing and training pipeline

---

## 🏗️ Architecture

```text
             T2 MRI
                │
                ▼
        ┌───────────────┐
        │ U-Net         │
        │ Generator     │
        └───────┬───────┘
                │
                ▼
        Synthetic T1ce
                │
       ┌────────┴────────┐
       │                 │
       ▼                 ▼
   Real T1ce       Generated T1ce
       │                 │
       └────────┬────────┘
                ▼
        PatchGAN Discriminator
                │
                ▼
            Real / Fake
```

The **Generator** learns to translate T2 MRI images into T1ce-like images while preserving important spatial information.

The **PatchGAN Discriminator** evaluates local image regions to encourage realistic textures and structures.

---

## 📂 Project Structure

```text
NEUROGAN/
│
├── data/
│   ├── raw/
│   │   └── Raw MRI / NIfTI files
│   │
│   └── processed/
│       └── Processed MRI files / slices
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── results/
│   └── samples/
│       └── Generated MRI samples
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── .gitignore
├── environment.yml
├── gpu_checker.py
├── run_train.sh
└── README.md
```

### Main Components

| Component           | Purpose                      |
| ------------------- | ---------------------------- |
| `data/raw/`         | Raw MRI/NIfTI dataset        |
| `data/processed/`   | Preprocessed MRI data        |
| `notebooks/`        | Dataset exploration          |
| `src/preprocess.py` | Data preprocessing           |
| `src/train.py`      | GAN training                 |
| `src/evaluate.py`   | Model evaluation             |
| `results/samples/`  | Generated MRI samples        |
| `gpu_checker.py`    | GPU availability check       |
| `run_train.sh`      | Training pipeline automation |
| `environment.yml`   | Project dependencies         |

---

## ⚙️ Tech Stack

* **Python**
* **TensorFlow**
* **NumPy**
* **Jupyter Notebook**
* **Conda**
* **NIfTI**
* **CUDA/GPU**
* **Bash**

### Deep Learning

* Conditional GAN
* Pix2Pix
* U-Net
* PatchGAN
* Convolutional Neural Networks

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/INDIAN123456890/NEUROGAN.git
cd NEUROGAN
```

### 2. Create Environment

```bash
conda env create -f environment.yml
conda activate synthetic_mri_gan
```

### 3. Check GPU

```bash
python gpu_checker.py
```

### 4. Preprocess Data

Place the raw MRI data inside:

```text
data/raw/
```

Then run:

```bash
python src/preprocess.py
```

Processed files will be generated inside:

```text
data/processed/
```

### 5. Train the Model

```bash
bash run_train.sh
```

Or directly:

```bash
python src/train.py
```

### 6. Evaluate

```bash
python src/evaluate.py
```

Generated samples are stored in:

```text
results/samples/
```

---

## 📊 Evaluation

The generated MRI images are evaluated using:

| Metric   | Purpose                                     |
| -------- | ------------------------------------------- |
| **PSNR** | Measures pixel-level reconstruction quality |
| **SSIM** | Measures structural similarity              |

Both metrics are used alongside visual inspection to assess the generated images.

---

## 🔄 Training Pipeline

```text
Raw NIfTI MRI
      ↓
Preprocessing
      ↓
Paired T2 + T1ce Dataset
      ↓
Pix2Pix GAN Training
      ↓
Checkpoint Saving
      ↓
Synthetic T1ce Generation
      ↓
PSNR + SSIM Evaluation
```

---

## ⚠️ Limitations

* GAN training can be computationally intensive and unstable.
* Generated images may contain artifacts.
* Results depend heavily on dataset quality and modality alignment.
* PSNR and SSIM alone cannot determine clinical usefulness.
* Synthetic MRI images should not replace actual clinical MRI scans.

---

## 🔮 Future Improvements

* 3D MRI GAN architecture
* Attention-based U-Net
* Improved GAN loss functions
* Multi-modal MRI translation
* Diffusion-based image synthesis
* Additional perceptual evaluation metrics
* Experiment tracking and hyperparameter optimization

---

## 👨‍💻 Author

**Sahil Salunke**

B.E. CSE (Data Science)

Interested in **Data Engineering, Data Science, Machine Learning, Deep Learning, Generative AI, and Computer Vision**.

---

## ⭐ Support

If you find **NEUROGAN** useful or interesting, consider giving the repository a ⭐ on GitHub.

**Live Neurogan Website:**
https://neurogan-puce.vercel.app/

**Repository:**
https://github.com/INDIAN123456890/NEUROGAN
