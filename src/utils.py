# src/utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
from src.config import Config
import imageio

os.makedirs(Config.SAMPLE_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

def denormalize(img):
    # img in [-1,1] -> [0,1]
    return (img + 1.0) / 2.0

def save_sample_images(generator, input_batch, target_batch, epoch, n=3, sample_dir=Config.SAMPLE_DIR):
    # input_batch, target_batch are tensors (B, H, W, 1)
    pred = generator(input_batch, training=False)
    input_np = input_batch.numpy()
    target_np = target_batch.numpy()
    pred_np = pred.numpy()
    for i in range(min(n, input_np.shape[0])):
        inp = denormalize(input_np[i].squeeze())
        tar = denormalize(target_np[i].squeeze())
        pr = denormalize(pred_np[i].squeeze())
        out = np.hstack([inp, pr, tar])
        out = (out * 255).astype('uint8')
        fname = os.path.join(sample_dir, f"epoch{epoch:03d}_sample{i}.png")
        imageio.imwrite(fname, out)
