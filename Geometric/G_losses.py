import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
K = keras.backend

def BCE_loss(lbl, pred):
    return K.mean(keras.losses.binary_crossentropy(lbl, pred))

def acc(lbl, pred):
    return K.mean(K.equal(K.argmax(lbl, axis=-1), K.argmax(pred, axis=-1)))

def MSE_loss(lbl, pred):
    return K.mean(keras.losses.mse(lbl, pred))

def one_sided_label_smoothing(labels):
    return np.where(labels == 1., 9., labels)

def L1_norm(effect_map):
    return tf.reduce_mean(tf.abs(effect_map))

def L2_norm(effect_map):
    return tf.sqrt(tf.reduce_mean(tf.square(effect_map)))

def discriminator_loss(real, fake):
    real_loss = MSE_loss(tf.ones_like(real), real)
    fake_loss = MSE_loss(tf.zeros_like(fake), fake)
    return real_loss + fake_loss

def generator_loss(target_c, fake):
    y_gen = np.expand_dims(np.ones(target_c.shape[0]).astype("float32"), axis=-1)
    return MSE_loss(y_gen, fake)

def cycle_loss(input, like_input):
    return tf.reduce_mean(tf.abs(like_input - input))