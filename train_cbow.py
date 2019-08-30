# coding: utf-8
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Lambda, Input, Reshape, Dot, Flatten
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard

import pickle
import numpy as np

from util import create_contexts_target, most_similar
from dataset import ptb
from negative_sampling import generate_with_negative_sample

tensorboard_callback = TensorBoard(log_dir='logs/cbow')

window_size = 10
hidden_size = 100
batch_size = 100
max_epoch = 15
sample_size = 5

corpus, word_to_id, id_to_word = ptb.load_data('train')
test_corpus = ptb.load_data('test')[0]

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
test_contexts, test_target = create_contexts_target(test_corpus, window_size)

contexts_input = Input(shape=(window_size * 2,), name='contexts_input')
target_input = Input(shape=(1,), name='target_input')

embed = Embedding(vocab_size, hidden_size, input_length=window_size * 2)

contexts_embed = embed(contexts_input)
contexts_hidden = Lambda(lambda arr: K.mean(arr, axis=1))(contexts_embed)

target_embed = Embedding(vocab_size, hidden_size, input_length=1)(target_input)
target_hidden = Reshape((hidden_size, ))(target_embed)

embed_dot = Dot(axes=1)([contexts_hidden, target_hidden])
output = Dense(1, activation='sigmoid')(embed_dot)

model = Model(inputs=[contexts_input, target_input], outputs=output)
# print('corpus', corpus.shape)
print('contexts', contexts.shape)
print('target', target.shape)

model.compile(
    optimizer='Adam',
    loss='binary_crossentropy',
    metrics=['acc']
)

print(model.summary())
hist = model.fit_generator(
    generate_with_negative_sample(
        corpus, contexts, target, batch_size, sample_size=sample_size),
    steps_per_epoch=len(contexts) // batch_size,
    initial_epoch=0,
    epochs=max_epoch, callbacks=[tensorboard_callback],
)

word_vecs = model.get_weights()[0]
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'  # or 'skipgram_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)
"""
"""
