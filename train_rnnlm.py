# coding: utf-8
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Lambda, Input, Reshape, Dot, Flatten, LSTM, TimeDistributed, Reshape
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam,RMSprop
from tensorflow.keras.losses import sparse_categorical_crossentropy,categorical_crossentropy

import pickle
import numpy as np
import os
from dataset import ptb
import math
import shutil
# ハイパーパラメータの設定
batch_size = 20
wordvec_size = 100
hidden_size = 100  # RNNの隠れ状態ベクトルの要素数
time_size = 35  # RNNを展開するサイズ
max_epoch = 40
max_grad = 0.25

save_weights_path='rrlm_w.h5'
checkpoint_dir = './training_checkpoints'
tensorboard_dir='logs/rrlm'

def generate_batch(x, t, batch_size, time_size):
    time_idx = 0
    while True:
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')

        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]  # バッチの各サンプルの読み込み開始位置

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + time_idx) % data_size]
                batch_t[i, time] = t[(offset + time_idx) % data_size]
            time_idx += 1
        yield batch_x, batch_t


# 学習データの読み込み
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_val, _, _ = ptb.load_data('val')
corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]
xs_val = corpus_val[:-1]
ts_val = corpus_val[1:]


gen = generate_batch(xs, ts, batch_size, time_size)


def build_model(batch_size):
    emb=Embedding(vocab_size, wordvec_size,batch_input_shape=(batch_size,None))
    model = Sequential([
        emb,
        LSTM(hidden_size,
          return_sequences=True,
          stateful=True,
          dropout=0.5
        ),
        LSTM(hidden_size,
          return_sequences=True,
          stateful=True,
          dropout=0.5
        ),
        LSTM(hidden_size,
          return_sequences=True,
          stateful=True,
          dropout=0.5
        ),
        LSTM(hidden_size,
          return_sequences=True,
          stateful=True,
          dropout=0.5
        ),
        LSTM(hidden_size,
          return_sequences=True,
          stateful=True,
          dropout=0.5
        ),
        Dense(vocab_size)
    ])
    return model
    
def loss(labels, logits):
    return sparse_categorical_crossentropy(labels, logits, from_logits=True)
    
def ppl(y_true, y_pred):
    loss = sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    perplexity = K.cast(K.pow(math.e, K.mean(loss, axis=-1)), K.floatx())
    return perplexity


if __name__ == '__main__':
    optimizer = RMSprop(clipnorm=max_grad)
    model=build_model(batch_size)

    model.load_weights(save_weights_path)


    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[ppl]
    )

    print(model.summary())

    # Directory where the checkpoints will be saved
    # Name of the checkpoint files
    
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        
    if os.path.exists(tensorboard_dir):
        shutil.rmtree(tensorboard_dir)
    
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    early_stop=EarlyStopping(monitor='val_ppl',patience=5)
    tensorboard_callback = TensorBoard(log_dir=tensorboard_dir)
    reduce_lr=ReduceLROnPlateau(monitor='val_ppl',factor=0.1,patience=0,)

    model.fit_generator(
        generate_batch(xs, ts, batch_size, time_size),
        steps_per_epoch=len(xs) // (batch_size * time_size),
        validation_data=generate_batch(xs_val,ts_val,batch_size,time_size),
        validation_steps=len(xs_val)//(batch_size*time_size),
        epochs=max_epoch,
        callbacks=[tensorboard_callback,checkpoint_callback,reduce_lr,early_stop],
        verbose=1
    )

    model.save_weights(save_weights_path)


