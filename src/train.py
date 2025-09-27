# src/train.py
import os
import time
import tensorflow as tf
from src.models import Generator_UNet, Discriminator_PatchGAN
from src.losses import generator_loss, discriminator_loss
from src.data_loader import build_datasets
from src.utils import save_sample_images
from src.config import Config

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
print("VRAM Allocator set to cuda_malloc_async.")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU Memory Growth enabled.")
    except RuntimeError as e:
        print(e)

# create models
generator = Generator_UNet(input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, Config.CHANNELS))
discriminator = Discriminator_PatchGAN(input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, Config.CHANNELS))

# optimizers
gen_optimizer = tf.keras.optimizers.Adam(Config.LEARNING_RATE, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(Config.LEARNING_RATE, beta_1=0.5)

start_epoch = tf.Variable(1)

# Checkpoint manager
ckpt = tf.train.Checkpoint(generator=generator, discriminator=discriminator,
                           gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer,
                           start_epoch=start_epoch)

ckpt_manager = tf.train.CheckpointManager(ckpt, Config.CHECKPOINT_DIR, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    try:
        ckpt_filename = os.path.basename(ckpt_manager.latest_checkpoint)
        ckpt_number = int(ckpt_filename.split('-')[-1])

        true_start_epoch = ckpt_number + 1

        if start_epoch.numpy() < true_start_epoch:
            start_epoch.assign(true_start_epoch)

    except Exception as e:
        print(f"Warning: Could not parse checkpoint number. Resuming using saved variable. Error: {e}")

    print(f"Restored checkpoint from {ckpt_manager.latest_checkpoint}")
    print(f"Resuming training from Epoch {start_epoch.numpy()}")

# load datasets
train_ds, val_ds, test_ds = build_datasets(img_size=Config.IMG_SIZE, channels=Config.CHANNELS)

# training step
@tf.function
def train_step(input_image, target):
    with tf.GradientTape(persistent=True) as tape:
        gen_output = generator(input_image, training=True)

        disc_real = discriminator([input_image, target], training=True)
        disc_generated = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gan_loss, l1_loss = generator_loss(disc_generated, gen_output, target)
        disc_loss = discriminator_loss(disc_real, disc_generated)

    gradients_of_generator = tape.gradient(gen_total_loss, generator.trainable_variables)
    gradients_of_discriminator = tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_total_loss, disc_loss, gan_loss, l1_loss

def train(epochs=Config.EPOCHS):
    if train_ds is None:
        raise RuntimeError("No training dataset found. Run preprocess.py and ensure processed data exists.")

    for epoch in range(start_epoch.numpy(), epochs + 1):

        start = time.time()
        print(f"Epoch {epoch}/{epochs}")

        for step, (input_image, target) in enumerate(train_ds):
            g_loss, d_loss, gan_loss, l1_loss = train_step(input_image, target)
            if step % 100 == 0:
                print(f"  step {step}: g_loss={g_loss:.4f}, d_loss={d_loss:.4f}")

        start_epoch.assign_add(1)
        ckpt_manager.save()

        # save sample images
        if Config.SAVE_SAMPLE_EVERY and epoch % Config.SAVE_SAMPLE_EVERY == 0:
            sample_batch = None
            try:
                sample_batch = next(iter(val_ds))
            except Exception:
                sample_batch = next(iter(train_ds))
            save_sample_images(generator, sample_batch[0], sample_batch[1], epoch)

        print(f"Time for epoch {epoch} is {time.time() - start:.2f} sec")
    print("Training finished.")


if __name__ == "__main__":
    train()