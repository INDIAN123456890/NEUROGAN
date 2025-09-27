# src/config.py
import os

class Config:
    # Paths
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RAW_DIR = os.path.join(ROOT, "data", "raw")
    PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
    CHECKPOINT_DIR = os.path.join(ROOT, "results", "checkpoints")
    SAMPLE_DIR = os.path.join(ROOT, "results", "samples")
    METRICS_FILE = os.path.join(ROOT, "results", "metrics.json")

    # Image settings
    IMG_SIZE = 256
    CHANNELS = 1  # single-channel MRI slices

    # Training
    BATCH_SIZE = 4
    EPOCHS = 10
    LEARNING_RATE = 2e-4
    LAMBDA_L1 = 100.0  # weight for L1 loss

    # Random seed
    SEED = 42

    # Other
    SAVE_SAMPLE_EVERY = 1  # epochs
