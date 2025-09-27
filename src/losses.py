# src/losses.py
import tensorflow as tf
from src.config import Config

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = bce(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = bce(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = (real_loss + generated_loss) * 0.5
    return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target, LAMBDA=Config.LAMBDA_L1):
    gan_loss = bce(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss
