# src/models.py
import tensorflow as tf
from tensorflow.keras import layers

initializer = tf.random_normal_initializer(0., 0.02)

def downsample(filters, size, apply_batchnorm=True):
    result = tf.keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=not apply_batchnorm))
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    result = tf.keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result


def Generator_UNet(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)

    # ENCODER: 8 Downsample steps for 256x256 input
    d1 = downsample(64, 4, apply_batchnorm=False)(inputs)  # 128x128
    d2 = downsample(128, 4)(d1)  # 64x64
    d3 = downsample(256, 4)(d2)  # 32x32
    d4 = downsample(512, 4)(d3)  # 16x16
    d5 = downsample(512, 4)(d4)  # 8x8
    d6 = downsample(512, 4)(d5)  # 4x4
    d7 = downsample(512, 4)(d6)  # 2x2
    d8 = downsample(512, 4)(d7)  # 1x1

    # DECODER: 8 Upsample steps
    u1 = upsample(512, 4, apply_dropout=True)(d8)
    u1 = layers.Concatenate()([u1, d7])

    u2 = upsample(512, 4, apply_dropout=True)(u1)
    u2 = layers.Concatenate()([u2, d6])

    u3 = upsample(512, 4, apply_dropout=True)(u2)
    u3 = layers.Concatenate()([u3, d5])

    u4 = upsample(512, 4)(u3)
    u4 = layers.Concatenate()([u4, d4])

    u5 = upsample(256, 4)(u4)
    u5 = layers.Concatenate()([u5, d3])

    u6 = upsample(128, 4)(u5)
    u6 = layers.Concatenate()([u6, d2])

    u7 = upsample(64, 4)(u6)
    u7 = layers.Concatenate()([u7, d1])

    last = layers.Conv2DTranspose(1, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')
    outputs = last(u7)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="Generator")


def Discriminator_PatchGAN(input_shape=(256, 256, 1)):

    inp = layers.Input(shape=input_shape, name='input_image')
    tar = layers.Input(shape=input_shape, name='target_image')
    x = layers.Concatenate()([inp, tar])
    x = downsample(64, 4, apply_batchnorm=False)(x)  # 64x64
    x = downsample(128, 4)(x)  # 32x32
    x = downsample(256, 4)(x)  # 16x16
    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.ZeroPadding2D()(x)
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)  # patch output
    out = last(x)
    return tf.keras.Model(inputs=[inp, tar], outputs=out, name="Discriminator")
