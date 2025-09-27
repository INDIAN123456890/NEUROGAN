# src/preprocess.py
import os
from glob import glob
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
from src.config import Config

IMG_SIZE = Config.IMG_SIZE
RAW_DIR = Config.RAW_DIR
PROCESSED_DIR = Config.PROCESSED_DIR
os.makedirs(PROCESSED_DIR, exist_ok=True)

def normalize_slice(img):
    img = img.astype('float32')
    mean = img.mean()
    std = img.std() + 1e-8
    img = (img - mean) / std
    img = np.clip(img, -5, 5)  # remove outliers
    # min-max to [0,1], then to [-1,1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = img * 2.0 - 1.0
    return img

def find_subject_ids(raw_dir=RAW_DIR):

    subject_dirs = sorted([d for d in os.listdir(raw_dir)
                           if os.path.isdir(os.path.join(raw_dir, d))])

    subject_ids = [d for d in subject_dirs if not d.startswith('.')]

    return subject_ids


def process_subject(subject_id):
    subject_dir = os.path.join(RAW_DIR, subject_id)

    def find_file_for_suffix(suffixes):
        for suf in suffixes:
            patterns = [f"{subject_id}{suf}.nii.gz", f"{subject_id}{suf}.nii"]
            for p in patterns:
                candidate = os.path.join(subject_dir, p)
                if os.path.exists(candidate):
                    return candidate
        for f in glob(os.path.join(subject_dir, f"{subject_id}*")):
            if any(s in f.lower() for s in ["t2", "t1ce", "t1gd"]):
                suf_check = [s.strip('_').strip('-') for s in suffixes]
                file_name = os.path.basename(f).lower()
                if any(mod in file_name for mod in suf_check):
                    return f
        return None

    t2_path = find_file_for_suffix(["_t2", "-t2", "_T2", "-T2", "_t2w"])
    t1ce_path = find_file_for_suffix(["_t1ce", "_t1gd", "-t1ce", "-t1gd", "_t1ce_flair", "_t1ce"])

    if t2_path is None or t1ce_path is None:
        print(f"[skip] missing modality for {subject_id}. t2: {t2_path}, t1ce: {t1ce_path}")
        return

    t2_vol = nib.load(t2_path).get_fdata()
    t1gd_vol = nib.load(t1ce_path).get_fdata()

    min_slices = min(t2_vol.shape[2], t1gd_vol.shape[2])
    pair_slices = []
    for i in range(min_slices):
        # Extract and resize slices
        t2_slice = cv2.resize(t2_vol[:, :, i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        t1_slice = cv2.resize(t1gd_vol[:, :, i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        # Normalize
        t2_n = normalize_slice(t2_slice)
        t1_n = normalize_slice(t1_slice)

        pair_slices.append([t2_n, t1_n])

    arr = np.array(pair_slices, dtype=np.float32)
    save_path = os.path.join(PROCESSED_DIR, f"{subject_id}.npy")
    np.save(save_path, arr)
    print(f"[saved] {save_path} -> {arr.shape[0]} slices")

def main():
    subject_ids = find_subject_ids(RAW_DIR)
    if not subject_ids:
        print("No T2 files found in", RAW_DIR)
        return
    print(f"Found {len(subject_ids)} subjects. Processing...")
    for sid in tqdm(subject_ids):
        process_subject(sid)
    print("Done preprocessing.")

if __name__ == "__main__":
    main()
