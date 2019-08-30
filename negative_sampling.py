import numpy as np
import tensorflow as tf
import collections


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):

        p = self.word_p.copy()
        target_idx = target
        p[target_idx] = 0
        p /= p.sum()
        negative_sample = np.random.choice(
            self.vocab_size, size=self.sample_size, replace=False, p=p)

        return negative_sample


def generate_with_negative_sample(corpus, contexts, target, batch_size, sample_size, power=0.75):
    sampler = UnigramSampler(corpus, power, sample_size)
    data_size = len(contexts)

    train_label = np.array(([1] + [0] * sample_size)
                           * batch_size).reshape(-1, 1)
    max_iters = data_size // batch_size

    while True:
        idx = np.random.permutation(np.arange(data_size))

        contexts = contexts[idx]
        target = target[idx]

        for iters in range(max_iters):
            batch_context = contexts[iters *
                                     batch_size:(iters + 1) * batch_size]
            batch_context = np.repeat(batch_context, sample_size + 1, axis=0)

            batch_target = target[iters * batch_size:(iters + 1) * batch_size]

            ns = np.apply_along_axis(
                sampler.get_negative_sample, 1, batch_target)

            batch_target = np.concatenate(
                (batch_target, ns), axis=1).reshape((-1, 1))

            yield [batch_context, batch_target], train_label
