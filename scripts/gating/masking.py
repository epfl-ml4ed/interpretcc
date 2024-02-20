# Code inspired and adapted from https://github.com/thomasverelst/dynconv

import tensorflow as tf
from tensorflow.keras.layers import Masking, Bidirectional, LSTM
from tensorflow.math import log
from tensorflow.random import uniform

class Mask():
    '''Holds mask properties'''

    def __init__(self, hard, soft):
        self.hard = hard
        self.active = tf.reduce_sum(hard, axis=-1)
        self.total = hard.shape[-1]
        self.soft = soft

    def size(self):
        return self.hard.shape

    def __repr__(self):
        return f'Mask with {self.active}/{self.total} positions'


class MaskUnit(tf.Module):
    ''' Generates the mask and applies the gumbel softmax trick'''
    
    def __init__(self, n_groups):
        super(MaskUnit, self).__init__()
        self.rnn_mask = Masking(mask_value=-1.)
        self.masker = Bidirectional(LSTM(n_groups, activation='sigmoid'), merge_mode='ave')
        self.gumbel = Gumbel()

    def __call__(self, x, meta, training=True):
        x = self.rnn_mask(x)
        soft = self.masker(x)
        hard = self.gumbel(soft, meta['gumbel_temp'], meta['gumbel_noise'], training)
        mask = Mask(hard, soft)
        meta['mask'] = mask
        return mask


class Gumbel(tf.Module):
    '''
    For differentiable discrete outputs. Applies Gumbel-Softmax trick on every element of input. 
    '''

    def __init__(self, eps=1e-8):
        super(Gumbel, self).__init__()
        self.eps = eps

    def __call__(self, x, temp=1., noise=True, training=True):
        if not training: # No Gumbel noise during inference
            return tf.cast(x >= 0.5, tf.float32)
        
        if noise:
            eps = self.eps
            in_shape = x.shape
            U1, U2 = uniform(in_shape), uniform(in_shape)
            g1, g2 = -log(-log(U1 + eps)+eps), -log(-log(U2 + eps)+eps)
            x = x + g1 - g2

        soft = tf.sigmoid(x / temp)
        hard = tf.stop_gradient(tf.cast(soft >= 0.5, tf.float32) - soft) + soft
        return hard