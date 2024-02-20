import ast
import numpy as np
import tensorflow as tf

from scripts.nets import Baseline as BiLSTMSigmoid
from .routing_modules import Classifier, Selector

# Example and default configuration
default_config = {
    'feature_sets': [
        [0, 1, 2, 3],
        [4, 5, 6, 7], 
        [0, 2, 4, 6],
        [1, 3, 5, 7]
    ],
    'classifier': {
        'dense': True,
        'units': 64,
        'layers': 2
    }
}

class MultiPathwaysNet(tf.keras.Model):
    '''Pathways network grouping together chosen features'''

    def __init__(self, config=default_config):
        ''' MultiPathwaysNet model initialization

        Args:
            config: Configuration (confidence config and easy set)
        '''
        super(MultiPathwaysNet, self).__init__()

        # Check for correct configuration
        assert 'feature_sets' in config
        assert 'classifier' in config
        
        # Useful values
        n_groups = len(config['feature_sets'])
        
        # Classification layer to choose path
        self.classifier = Classifier(n_out=n_groups, config=config['classifier'])

        # Feature selectors
        selectors = [Selector(feature_set=f_set) for f_set in config['feature_sets']]

        # Prediction layers
        predictors = [BiLSTMSigmoid(n_units=32, rec_layers=2) for _ in selectors]

        # Attach feature selector to predictors
        self.pathways = [(s, p) for s, p in zip(selectors, predictors)]

    
    def call(self, input_, *args, **kwargs):
        '''MultiPathwaysNet forward pass'''
        path_choice = self.classifier(input_)

        predictions = [predictor(selector(input_)) for selector, predictor in self.pathways]

        output = path_choice * tf.concat(predictions, axis=-1)
        return tf.reduce_sum(output, axis=-1)

    def classification_confidence(self, input_):
        '''Returns classification softmax scores for given input'''
        return self.classifier.soft_classify(input_)

    def classify(self, input_):
        '''Returns classification for given input'''
        return self.classifier(input_)

def make_feature_sets(course, path, feature_types):
    '''Helper: Make feature type sets to pass to feature selector
    Assumption: feature_types in same order as from how your data was loaded
    
    Args:
        course: (str) Course name/identifier
        path: (str) Course data path
        feature_types: (List[str]) Feature types

    Returns:
        feature_sets
    '''
    i = 0
    feature_sets = []
    for feature_type in feature_types:
        names = open(f'{path}/eq_week-{feature_type}-{course}/settings.txt', 'r').read()
        names = ast.literal_eval(names)['feature_names']
        names = [f"{'_'.join(name.split('_')[:-1])} {name.split('_')[-1].split(' ')[1]}" if 'function ' in name else name for name in names]

        f_set = []
        for n in names:
            f_set.append(i)
            i += 1
        feature_sets.append(f_set)

    return feature_sets

def custom_feature_sets(name_sets, names):
    '''Make a custom list of feature sets given feature names
    
    Args:
        name_sets: (List[List[str]]) Feature sets with names
        names: (List[str]) Feature names in order of data

    Returns:
        Feature sets usable by Selector module
    '''
    feature_sets = []
    for s in name_sets:
        new_set = [names.index(n) for n in s]
        feature_sets.append(new_set)
    return feature_sets