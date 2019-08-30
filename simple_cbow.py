import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Layer, Dot
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential


class SimpleCBOW(Layer):
    def __init__(self, vocab_size, hidden_size):
        super(SimpleCBOW, self).__init__()
        self.V, self.H = vocab_size, hidden_size
        print((self.V, self.H))

    def build(self, input_shape):
        self.W_in = self.add_weight(
            shape=[self.V, self.H],
            initializer='random_normal',
            trainable=True,
            dtype=tf.float32,
        )
        self.W_out = self.add_weight(
            shape=[self.H, self.V],
            initializer='random_normal',
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)

    def call(self, contexts):
        self.h0 = tf.matmul(tf.cast(contexts[:, 0], tf.float32), self.W_in)
        self.h1 = tf.matmul(tf.cast(contexts[:, 1], tf.float32), self.W_in)
        self.h = ((self.h0 + self.h1) * 0.5)
        self.score = tf.matmul(self.h, self.W_out)
        return self.score
