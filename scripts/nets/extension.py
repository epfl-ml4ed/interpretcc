import tensorflow as tf

from scripts.gating import Gating, OutputWithMask
from scripts.routing import Selector, Classifier, BiLSTMBase
from .baseline import Baseline as BiLSTMSigmoid

class RoutingGatingModel(tf.keras.Model):
    '''Routing + Gating'''

    def __init__(self, config):
        '''RoutingGatingModel initialization'''
        super(RoutingGatingModel, self).__init__()

        # Check for correct configuration
        assert 'feature_sets' in config
        assert 'regularizor' in config
        
        n_groups = len(config['feature_sets'])

        # Gating layer to choose paths
        self.gating = Gating(n_groups, config['regularizor'])

        # Feature selectors
        selectors = [Selector(feature_set=f_set) for f_set in config['feature_sets']]

        # Prediction layers
        predictors = [BiLSTMSigmoid(n_units=32, rec_layers=2) for _ in selectors]

        # Attach feature selector to predictors
        self.pathways = [(s, p) for s, p in zip(selectors, predictors)]

        # Output layer
        self.out_layer = OutputWithMask()

    def call(self, input_):
        '''RoutingGatingModel forward pass'''
        path_choices = self.gating(input_)

        predictions = [predictor(selector(input_)) for selector, predictor in self.pathways]
        return self.out_layer(tf.concat(predictions, axis=-1), mask=path_choices)

    def get_callback(self):
        return self.gating.gumbel.activity_regularizer

    def get_gates(self, x):
        return self.gating(x)


# Default config for RoutingPreFreeze
default_chooser_config = {
    'dense': False,
    'layers': 2,
    'units': 32
}

class RoutingPreFreeze(tf.keras.Model):
    '''Routing model with pretrained output layers'''
    def __init__(self, pretrained_ls, chooser_config=default_chooser_config):
        ''' Initialization
        
        Args:
            pretrained_ls: List of pretrained sub-networks
            chooser_config: Configuration for the "choice" classification part of the network (has default config)
        '''
        super(RoutingPreFreeze, self).__init__()

        n = len(pretrained_ls)

        self.chooser = Classifier(n, chooser_config)

        self.pathways = pretrained_ls
        # Freeze sub-networks
        for pathway in self.pathways:
            pathway.trainable = False

    def call(self, input_):
        '''Forward pass'''
        path_choice = self.chooser(input_)
        predictions = [p(input_) for p in self.pathways]
        output = path_choice * tf.concat(predictions, axis=-1)
        return tf.reduce_sum(output, axis=-1)

    def classify(self, input_):
        '''Returns classification for given input'''
        return self.chooser(input_)

def make_pathway(in_shape, feature_set):
    '''Makes a subnetwork for the RoutingPreFreeze
    
    Args:
        in_shape: Input shape without batch dimension
        feature_set: Feature set to select
    '''
    input_layer = tf.keras.Input(shape=in_shape)
    selector = Selector(feature_set=feature_set)(input_layer)
    final_output = BiLSTMSigmoid(n_units=32, rec_layers=2)(selector)
    model = tf.keras.Model(inputs=input_layer, outputs=final_output)
    return model

def make_pathways(in_shape, feature_sets):
    '''Returns list of sub-networks for RoutingPreFreeze
    
    Args:
        in_shape: Input shape without batch dimension
        feature_sets: Feature sets (e.g. output of make_feature_sets)
    '''
    return [make_pathway(in_shape, fset) for fset in feature_sets]


class WeightedRouting(tf.keras.Model):
    '''Output predictions are weighted'''

    def __init__(self, feature_sets):
        ''' Model initialization

        Args:
            feature_sets: Feature sets
        '''
        super(WeightedRouting, self).__init__()
        
        # Useful values
        n_groups = len(feature_sets)
        
        # Layers for path weights
        self.base = BiLSTMBase(config={'units':64, 'layers':2})
        self.denseSM = tf.keras.layers.Dense(n_groups, activation='softmax')

        # Feature selectors
        selectors = [Selector(feature_set=f_set) for f_set in feature_sets]

        # Prediction layers
        predictors = [BiLSTMSigmoid(n_units=32, rec_layers=2) for _ in selectors]

        # Attach feature selector to predictors
        self.pathways = [(s, p) for s, p in zip(selectors, predictors)]

    
    def call(self, input_):
        '''Forward pass'''
        sm = self.denseSM(self.base(input_))

        predictions = [predictor(selector(input_)) for selector, predictor in self.pathways]

        output = sm * tf.concat(predictions, axis=-1)
        return tf.reduce_sum(output, axis=-1)

    def route_weightings(self, input_):
        return self.denseSM(self.base(input_))

class WeightedThresholdRouting(tf.keras.Model):
    '''
    Weighted Routing 
    with routes activated only if corresponding softmax layer entry is above threshold
    '''
    def __init__(self,feature_sets, threshold=0.05, pretrained_ls=None):
        '''
        Args:
            feature_sets: Feature sets
            threshold: Confidence threshold (default = 0.05)
            pretrained_ls: Optional pretrained expert sub-networks (default = None)
        '''
        assert threshold >= 0.0 and threshold <= 1.0
        super(WeightedThresholdRouting, self).__init__()
        
        self.threshold = tf.constant(threshold)

        n_groups = len(feature_sets)

        # Get route confidence with softmax
        self.base = BiLSTMBase(config={'units':64, 'layers':2})
        self.denseSM = tf.keras.layers.Dense(n_groups, activation='softmax')

        if pretrained_ls is not None:
            self.pretrained = True
            self.pathways = pretrained_ls
            # Freeze sub-networks (optionally make this an option in initialization to freeze or not)
            for pathway in self.pathways:
                pathway.trainable = False
        else:
            self.pretrained = False
            selectors = [Selector(feature_set=f_set) for f_set in feature_sets]
            predictors = [BiLSTMSigmoid(n_units=32, rec_layers=2) for s in selectors]
            self.pathways = [(s, p) for s, p in zip(selectors, predictors)]


    def call(self, input_):
        '''Forward pass'''
        # Softmax for confidence
        sm = self.denseSM(self.base(input_))
        # Differentiable mask
        mask = tf.stop_gradient(tf.cast(sm >= self.threshold, tf.float32) - sm) + sm
        # Expert sub-networks predictions
        if self.pretrained:
            predictions = [subnet(input_) for subnet in self.pathways]
        else:
            predictions = [p(s(input_)) for s, p in self.pathways]
        # Nullify weights that are not >= threshold
        ws = sm * mask
        # Multiply weight by prediction and sum all weighted predictions
        w_pred = tf.reduce_sum(ws * tf.concat(predictions, axis=-1), axis=-1)
        # Divide by total weight (or 1 if sum of weights = 0)
        return w_pred / tf.maximum(1., tf.reduce_sum(ws, axis=-1))

    def route_weights(self, input_):
        '''Returns route confidence scores'''
        return self.denseSM(self.base(input_))

    def pathway_pred(self, path_id, x_test):
        '''Returns chosen pathway prediction on input data'''
        assert path_id >= 0 and path_id < len(self.pathways)
        
        # Selector and predictor for pathway
        s, p = self.pathways[path_id]

        return p(s(x_test))

    def route_choices(self, input_):
        '''Returns the route choices for the given input'''
        sm = self.denseSM(self.base(input_))
        return tf.cast(sm >= self.threshold, tf.float32)

