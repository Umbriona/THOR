import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Activation, Embedding, Input, LeakyReLU, LayerNormalization, BatchNormalization, Softmax, Dropout, Concatenate
from utils.layers_new import GumbelSoftmax


def downsample(filters, size, norm="instance"):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv1D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

    if norm=="batch":
        result.add(tf.keras.layers.BatchNormalization())
    elif norm=="instance":
        result.add(tfa.layers.InstanceNormalization())
        
    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False, norm="instance"):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv1DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    if norm=="batch":
        result.add(tf.keras.layers.BatchNormalization())
    elif norm=="instance":
        result.add(tfa.layers.InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[512, 21])

    down_stack = [
    downsample(64, 4, norm=None),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv1DTranspose(21, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='linear')  # (batch_size, 256, 256, 3)
    gsm_out = GumbelSoftmax()

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    x= gsm_out(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator(norm="instance"):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[ 512, 21], name='input_image')

    down1 = downsample(64, 4, False)(inp)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    conv = tf.keras.layers.Conv1D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(down3)  # (batch_size, 31, 31, 512)

    if norm=="batch":
        batchnorm1 =  tf.keras.layers.BatchNormalization()(conv)
    elif norm=="instance":
        batchnorm1 = tfa.layers.InstanceNormalization()(conv)


    #leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    

    last = tf.keras.layers.Conv1D(21, 4, strides=1,
                                kernel_initializer=initializer,
                                 activation="sigmoid")(batchnorm1)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)