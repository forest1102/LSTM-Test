# coding: utf-8
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Lambda, Input, Reshape, Dot, Flatten, LSTM, TimeDistributed, Reshape
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

import pickle
import numpy as np
import os
from dataset import ptb
import math

# ハイパーパラメータの設定
batch_size = 20
wordvec_size = 100
hidden_size = 100  # RNNの隠れ状態ベクトルの要素数
time_size = 35  # RNNを展開するサイズ
lr = 20.0
max_epoch = 4
max_grad = 0.25

input = Input(batch_shape=(batch_size, None))
output = Embedding(vocab_size, wordvec_size)(input)
output = LSTM(hidden_size,
              return_sequences=True,
              stateful=True,
              )(output)
output = Dense(vocab_size)(output)

model = Model(input, output)
