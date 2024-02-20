import tensorflow as tf

from scripts.nets import Baseline as BiLSTMSigmoid
from .routing_modules import Confidence, Selector

# Example and default configuration
default_config = {
    'confidence': {
        'dense': True,
        'units': 64,
        'layers': 2
    },
    'easy_set': [ 0,  7, 16, 17, 23, 29, 33, 36, 38, 41, 43, 44]
}

class EasyHardNet(tf.keras.Model):
    '''Hard-easy architecture'''
    
    def __init__(self, config=default_config):
        ''' EasyHard model initialization

        Args:
            config: Configuration (confidence config and easy set)
        '''
        super(EasyHardNet, self).__init__()
        # TODO: maybe add asserts for configuration (e.g. assert 'confidence' in config)
        self.confidence = Confidence(config['confidence'])
        
        # Easy path => truncate to smaller feature set
        self.selector = Selector(config['easy_set'])
        self.easy_predict = BiLSTMSigmoid(n_units=32, rec_layers=2)
        
        # Hard path => keep all features
        self.hard_predict = BiLSTMSigmoid(n_units=64, rec_layers=2)

    def call(self, input_, *args, **kwds):
        confidence = self.confidence(input_)

        easy = self.easy_predict(self.selector(input_))
        hard = self.hard_predict(input_)

        output = confidence * tf.concat([easy, hard], axis=-1)
        return tf.reduce_sum(output, axis=-1)

    def confidence_scores(self, input_):
        '''Returns confidence scores for the given input'''
        return self.confidence.confidence_scores(input_)

    def path_choices(self, input_):
        '''Returns path choices for the given input'''
        return self.confidence(input_)