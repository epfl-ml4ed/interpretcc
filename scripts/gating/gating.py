import tensorflow as tf

from collections.abc import Callable

from tensorflow.random import uniform
from tensorflow.math import log

from tensorflow import keras
from keras import layers, regularizers
from keras.layers import Bidirectional, LSTM, Dense
from keras.callbacks import LambdaCallback

from scripts.nets import Baseline as BiLSTMSigmoid

# ==== MODELS ====

class FeatureGatingModel(keras.Model):
    '''Custom Feature Gating model'''

    def __init__(self, n_gates, gate_units=8, gating_reg=regularizers.L1(1e-5)):
        '''Initialize GatingModel
        
        Args:
            n_gates: Number of features
            gate_units: Number of recurrent units for each gate BiLSTM (default = 8)
            gating_reg: Gating regularization (default = L1(1e-5))
        '''
        super().__init__(name='GatingModel')
        self.gating = Gating(n_gates, reg=gating_reg)
        self.predictors = [BiLSTMSigmoid(gate_units) for _ in range(n_gates)]
        self.out_layer = OutputWithMask()

    def call(self, inputs, *args, **kwargs):
        # Compute gating mask
        mask = self.gating(inputs)

        # Split features
        xs = tf.split(inputs, inputs.shape[-1], axis=-1)

        # Compute outputs
        predictions = [bilstm(in_) for in_, bilstm in zip(xs, self.predictors)]
        outs = tf.concat(predictions, axis=-1)

        return self.out_layer(outs, mask=mask)

    def get_callback(self):
        return self.gating.gumbel.activity_regularizer

    def get_mask(self, inputs):
        return self.gating(inputs)


# ==== LAYERS ====

class OutputWithMask(layers.Layer):
    '''Output layer (mask-consuming layer)'''

    def __init__(self):
        '''OutputWithMask layer initialization'''
        super().__init__(name='OutputWithMask')

    def call(self, inputs, mask=None):
        '''OutputWithMask forward pass'''
        return tf.reduce_sum(inputs * mask, axis=-1) / tf.maximum(tf.reduce_sum(mask, axis=-1), 1.)

class Gating(layers.Layer):
    '''Gating layer (mask-generating layer)'''
    def __init__(self, n_gates, reg):
        '''Gating layer initialization
        
        Args:
            n_gates: Number of gates
            reg: Activity regularizer
        '''
        super().__init__(name='Gating')
        self.rnn_mask = layers.Masking(mask_value=-1.)
        self.masker = Bidirectional(LSTM(n_gates), merge_mode='ave')
        self.gumbel = GumbelSigmoid(activity_regularizer=reg)

    def call(self, inputs, training=None):
        '''Gating layer forward pass'''
        x = self.rnn_mask(inputs)
        logits = self.masker(x)
        return self.gumbel(logits, training)

    def compute_mask(self, inputs, mask=None):
        '''Compute Gating layer mask'''
        return tf.cast(self.call(inputs), tf.bool)



class GumbelSigmoid(layers.Layer):
    '''Applies the Gumbel trick element-wise'''

    def __init__(self, eps=1e-8, temp=1., activity_regularizer=None):
        '''Gumbel Sigmoid layer initialization
        
        Args:
            eps: Epsilon (default = 1e-8)
            temp: Gumbel temperature (default = 1.)
        '''
        super().__init__(name='GumbelSigmoid', activity_regularizer=activity_regularizer)
        self.eps = eps
        self.temp = temp

    def call(self, logits, hard=True, training=None, *args, **kwargs):
        '''GumbelSigmoid forward pass
        
        Args:
            logits: Input logits
            hard: Return as quantized (default = True)
            training: Layer training argument (default = None)

        Returns:
            GumbelSigmoid forward pass in training or inference
        '''
        if training is not True:
            return tf.cast(logits >= 0.0, tf.float32)

        # Add gumbel noise to logits
        eps = self.eps
        in_shape = tf.shape(logits)
        U1, U2 = uniform(in_shape), uniform(in_shape)
        g1, g2 = -log(-log(U1 + eps)+eps), -log(-log(U2 + eps)+eps)
        logits = logits + g1 - g2

        soft = tf.sigmoid(logits / self.temp)
        if not hard:
            return soft
        
        quantized = tf.stop_gradient(tf.cast(soft >= 0.5, tf.float32) - soft) + soft
        return quantized

# ==== REGULARIZATION ====

def make_callback(anon_func: Callable[[int], None]) -> LambdaCallback:
    ''' Make Callback function (helper to use in case of using annealing regularization)

    Example: make_callback(lambda e: print(e)) 
    
    Args:
        anon_func: (Callable[[int], None]) Function which takes in epoch number

    Returns:
        LambdaCallback called at epoch beginning
    '''
    return LambdaCallback(on_epoch_begin = lambda epoch, logs : anon_func(epoch))

@tf.keras.utils.register_keras_serializable(package='Custom', name='l1_annealing')
class AnnealingL1(regularizers.Regularizer, keras.callbacks.Callback):
    '''Annealing regularization'''

    def __init__(self, n_epochs: int, l1=1e-5):
        ''' AnnealingRegularizor initialization
        
        Args:
            n_epochs: Number of training epochs
            l1: L1 parameter
        '''
        super(AnnealingL1, self).__init__()
        self.current_epoch = tf.Variable(1., dtype=tf.float32)
        self.n_epochs = tf.constant(n_epochs, dtype=tf.float32)
        self.reg = regularizers.L1(l1)

    def __call__(self, x):
        return (self.current_epoch / self.n_epochs) * self.reg(x)

    def on_epoch_begin(self, epoch, logs=None):
        self.set_current_epoch(epoch+1)

    def set_current_epoch(self, new_epoch: int) -> None:
        '''Set current epoch to new value'''
        self.current_epoch.assign(new_epoch)

    def get_progress(self):
        return self.current_epoch / self.n_epochs

@tf.keras.utils.register_keras_serializable(package='Custom', name='mse_annealing')
class AnnealingMSE(regularizers.Regularizer, keras.callbacks.Callback):
    '''Annealing regularization'''

    def __init__(self, n_gates:int, n_epochs: int, alpha=1., sparsity_target=0.5):
        ''' AnnealingRegularizor initialization
        
        Args:
            n_gates: Number of gates
            n_epochs: Number of training epochs
            sparsity_target: Sparsity target (between 0 and 1)
        '''
        super(AnnealingMSE, self).__init__()
        self.n_gates = tf.constant(n_gates, dtype=tf.float32)
        self.current_epoch = tf.Variable(1., dtype=tf.float32)
        self.n_epochs = tf.constant(n_epochs, dtype=tf.float32)
        self.sparsity_target = tf.constant(sparsity_target, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)

    def __call__(self, x):
        # Calculate percentage of active gates
        percent = tf.reduce_sum(x) / self.n_gates

        # Get epochs progress
        progress = self.get_progress()

        # Calculate regularization
        reg = tf.zeros_like(progress)
        upper_bound = 1. - progress * (1. - self.sparsity_target)
        lower_bound = progress * self.sparsity_target
        reg += tf.maximum(0., percent - upper_bound) ** 2
        reg += tf.maximum(0., lower_bound - percent) ** 2

        reg += (percent - self.sparsity_target) ** 2

        return self.alpha * reg

    def on_epoch_begin(self, epoch, logs=None):
        self.set_current_epoch(epoch+1)

    def set_current_epoch(self, new_epoch: int) -> None:
        '''Set current epoch to new value'''
        self.current_epoch.assign(new_epoch)

    def get_progress(self):
        return self.current_epoch / self.n_epochs