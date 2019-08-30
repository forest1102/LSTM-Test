# coding: utf-8
from tensorflow.keras.layers import Dense, Dropout, Softmax, Embedding, GlobalAveragePooling1D, Average, Lambda, Reshape, Dot, RepeatVector
from tensorflow.keras.models import Sequential
from simple_cbow import SimpleCBOW
import sys
import numpy as np
from util import preprocess, create_contexts_target, convert_one_hot, most_similar
import tensorflow.keras.backend as K

window_size = 2
hidden_size = 5
batch_size = 3
max_epoch = 100
sample_size = 2

text = 'You say goodbye and I say hello. You say hello and I say goodbye.'
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)

contexts = contexts[:batch_size]
target = target[:batch_size]

print('target:', np.shape(target))
print('contexts:', np.shape(contexts))

embed = Embedding(vocab_size, hidden_size)
contexts_embed = embed(contexts)
print(contexts_embed.shape)

contexts_average = Lambda(lambda arr: K.mean(arr, axis=1))(contexts_embed)

print(contexts_average.shape)

contexts_repeated = RepeatVector(sample_size)(contexts_average)

print(contexts_repeated)

print('target: ')

target_embed_arr = np.array([embed(target), embed(target)])
print(target_embed_arr[0].shape)

"""
embed_dot = Lambda(
    lambda t_arr: K.map_fn(
        lambda x: Dot(axes=1)([t_arr[0], x]), t_arr[1]))([contexts_average, target_embed_arr])
"""
embed_dot = Dot(axes=1, normalize=True)(
    [np.array([contexts_average, contexts_average]), target_embed_arr])
print(embed_dot)

output = Dense(1, activation='sigmoid')(embed_dot)
print(output)
