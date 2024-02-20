import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Masking, Bidirectional, LSTM
from tensorflow.math import log
from tensorflow.random import uniform

class Selector(tf.Module):
    '''Feature truncator/selector'''

    def __init__(self, feature_set):
        super(Selector, self).__init__()
        self.set = feature_set

    def __call__(self, input_):
        return tf.gather(input_, self.set, axis=-1)


class Classifier(tf.Module):
    '''Classifier module'''

    def __init__(self, n_out, config):
        '''Classifier initialization
        
        Args:
            n_out: Number of classes
            config: Classifier configuration
        '''
        super(Classifier, self).__init__()

        # Check for correct configuration
        assert 'dense' in config
        assert 'layers' in config
        assert 'units' in config
        
        # Define discriminator
        if config['dense']:
            self.discriminator = DenseBase(config)
        else:
            self.discriminator = BiLSTMBase(config)

        # Logits
        self.predict = Dense(n_out)

        # Gumbel SoftMax Trick Module
        self.gumbel = GumbelSoftMax()

    def __call__(self, input_):
        '''Classifier forward pass'''
        logits = self.predict(self.discriminator(input_))
        return self.gumbel(logits)

    def soft_classify(self, input_):
        '''Classification but with soft labels'''
        logits = self.predict(self.discriminator(input_))
        return self.gumbel(logits, hard=False, training=None)


class Confidence(tf.Module):
    '''Easy-hard confidence module'''

    def __init__(self, config):
        '''Confidence initialization
        
        Args:
            config: Confidence configuration (dense, layers, units)
        '''
        super(Confidence, self).__init__()
        # Confidence == Classifier with 2 output classes
        self.classifier = Classifier(2, config)

    def __call__(self, input_):
        '''Confidence forward pass'''
        return self.classifier(input_)

    def confidence_scores(self, input_):
        '''Returns confidence scores for input'''
        return self.classifier.soft_classify(input_)


class GumbelSoftMax(tf.keras.layers.Layer):
    '''Gumbel SoftMax'''
    
    def __init__(self, eps=1e-8, temp=1.):
        '''GumbelSoftMax initialization
        
        Args:
            eps: Epsilon
            temp: Gumbel temperature
        '''
        super(GumbelSoftMax, self).__init__()
        self.eps = tf.constant(eps)
        self.temp = tf.constant(temp)

    def call(self, logits, hard=True, training=None):
        '''Gumbel SoftMax Trick forward pass'''
        in_shape =  tf.shape(logits)

        # Generate gumbel noise samples and add to logits
        U = uniform(in_shape)
        g = -log(-log(U + self.eps) + self.eps)
        y = g + logits
        
        soft = tf.nn.softmax(y / self.temp)
        if not hard:
            return soft
        
        # Which element has highest probability?
        which = tf.argmax(soft, axis=-1)
        # One-hot encode choice of element
        one_hot = tf.one_hot(which, depth=in_shape[-1])

        return tf.stop_gradient(one_hot - soft) + soft


class DenseBase(tf.Module):
    '''Fully connected discriminator'''
    
    def __init__(self, config):
        '''
        Args:
            config: Configuration (units, layers)
        '''
        super(DenseBase, self).__init__()
        self.flat = Flatten()
        self.fc = [Dense(config['units'], activation='relu') for _ in range(config['layers'])]

    def __call__(self, input_):
        '''DenseBase forward pass'''
        x = self.flat(input_)
        for layer in self.fc:
            x = layer(x)
        return x

class BiLSTMBase(tf.Module):
    '''Fully connected discriminator'''
    
    def __init__(self, config):
        '''
        Args:
            config: Configuration (units, layers)
        '''
        super(BiLSTMBase, self).__init__()
        self.mask = Masking(mask_value=-1.)
        self.bilstm = [Bidirectional(LSTM(config['units'], return_sequences=True)) for _ in range(config['layers']-1)]
        self.bilstm.append(Bidirectional(LSTM(config['units'])))

    def __call__(self, input_):
        '''BiLSTMBase forward pass'''
        x = self.mask(input_)
        for layer in self.bilstm:
            x = layer(x)
        return x