# src/evaluate.py
import json
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn, structural_similarity as ssim_fn
from src.models import Generator_UNet
from src.data_loader import build_datasets
from src.config import Config
import tensorflow as tf
import os

generator = Generator_UNet(input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, Config.CHANNELS))
ckpt = tf.train.Checkpoint(generator=generator)
ckpt_manager = tf.train.CheckpointManager(ckpt, Config.CHECKPOINT_DIR, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print("Restored checkpoint:", ckpt_manager.latest_checkpoint)
else:
    print("No checkpoint found - evaluation will proceed with untrained generator (likely poor results).")

_, _, test_ds = build_datasets(img_size=Config.IMG_SIZE, channels=Config.CHANNELS)
if test_ds is None:
    raise RuntimeError("No test dataset. Run preprocess.py and ensure processed data exists.")

psnrs, ssims = [], []
for inp, tar in test_ds:
    pred = generator(inp, training=False).numpy()  # [-1,1]
    inp_np = inp.numpy()
    tar_np = tar.numpy()
    for i in range(pred.shape[0]):
        target_img = tar_np[i].squeeze()
        pred_img = pred[i].squeeze()
        try:
            p = psnr_fn(target_img, pred_img, data_range=2.0)
            s = ssim_fn(target_img, pred_img, data_range=2.0)
        except Exception:
            # fallback normalize to [0,1] then compute
            targ01 = (target_img + 1) / 2.0
            pred01 = (pred_img + 1) / 2.0
            p = psnr_fn(targ01, pred01, data_range=1.0)
            s = ssim_fn(targ01, pred01, data_range=1.0)
        psnrs.append(p)
        ssims.append(s)

metrics = {"psnr_mean": float(np.mean(psnrs)), "psnr_median": float(np.median(psnrs)),
           "ssim_mean": float(np.mean(ssims)), "ssim_median": float(np.median(ssims)),
           "n_samples": len(psnrs)}

os.makedirs(os.path.dirname(Config.METRICS_FILE), exist_ok=True)
with open(Config.METRICS_FILE, "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved metrics:", Config.METRICS_FILE)
print(metrics)
