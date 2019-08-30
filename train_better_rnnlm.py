# coding: utf-8
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Lambda, Input, Reshape, Dot, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizer import SGD
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard

import pickle
import numpy as np

from util import create_contexts_target, most_similar
from dataset import ptb

# ハイパーパラメータの設定
batch_size = 20
wordvec_size = 100
hidden_size = 100  # RNNの隠れ状態ベクトルの要素数
time_size = 35  # RNNを展開するサイズ
lr = 20.0
max_epoch = 4
max_grad = 0.25

corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

model = Sequential([

])
optimizer = SGD(lr=lr, clipnorm=clip_grads)
