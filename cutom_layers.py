import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Layer, Dot
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential


class Embedding(Layer):
    __init__(self, W):
        super(Embedding, self).__init__()
        self.params = [W]
        self.idx = None

    def call(self, x):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
