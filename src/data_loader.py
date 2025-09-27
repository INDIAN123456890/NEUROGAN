# src/data_loader.py

import os
import numpy as np
import tensorflow as tf
from glob import glob
from src.config import Config
import random

# Use the maximum number of logical threads for parallel I/O and processing
PARALLEL_THREADS = 16

def _load_npy_file(file_path):

    file_path_str = file_path.numpy().decode('utf-8')
    arr = np.load(file_path_str)
    t2 = arr[:, 0, ...]  # Input (T2)
    t1 = arr[:, 1, ...]  # Target (T1ce)

    X = t2[..., np.newaxis].astype('float32')  # (S, H, W, 1)
    Y = t1[..., np.newaxis].astype('float32')  # (S, H, W, 1)

    return X, Y

def build_datasets(img_size, channels, processed_dir=Config.PROCESSED_DIR, batch_size=Config.BATCH_SIZE,
                   split=(0.7, 0.15, 0.15),
                   shuffle_subjects=True, seed=Config.SEED):
    npy_files = sorted(glob(os.path.join(processed_dir, "*.npy")))
    if len(npy_files) == 0:
        raise FileNotFoundError(f"No .npy files found in {processed_dir}. Run preprocess.py first.")

    if shuffle_subjects:
        random.Random(seed).shuffle(npy_files)

    n = len(npy_files)
    n_train = int(split[0] * n)
    n_val = int(split[1] * n)
    train_files = npy_files[:n_train]
    val_files = npy_files[n_train:n_train + n_val]
    test_files = npy_files[n_train + n_val:]

    output_types = (tf.float32, tf.float32)
    output_shapes = (
        tf.TensorShape([None, img_size, img_size, channels]),
        tf.TensorShape([None, img_size, img_size, channels])
    )

    def files_to_dataset_optimized(file_list):
        ds_paths = tf.data.Dataset.from_tensor_slices(file_list)
        ds_paths = ds_paths.shuffle(buffer_size=len(file_list))

        def map_file_to_slices(file_path):
            X_slices, Y_slices = tf.py_function(
                _load_npy_file,
                [file_path],
                Tout=output_types
            )
            X_slices.set_shape(output_shapes[0])
            Y_slices.set_shape(output_shapes[1])

            return tf.data.Dataset.from_tensor_slices((X_slices, Y_slices))

        ds = ds_paths.interleave(
            map_func=map_file_to_slices,
            cycle_length=PARALLEL_THREADS,
            block_length=1,
            num_parallel_calls=PARALLEL_THREADS
        )

        ds = ds.shuffle(buffer_size=2048)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = files_to_dataset_optimized(train_files)
    val_ds = files_to_dataset_optimized(val_files)
    test_ds = files_to_dataset_optimized(test_files)
    return train_ds, val_ds, test_ds