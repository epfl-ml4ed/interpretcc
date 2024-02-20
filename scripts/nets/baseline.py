import tensorflow as tf

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Dense, Masking, Bidirectional, LSTM, Layer, Flatten, Concatenate

class Baseline(tf.Module):
    '''
    Baseline Bidirectional LSTM module
    '''
    def __init__(self, n_units=32, rec_layers=1, sigmoid=True, name='Baseline'):
        ''' Baseline BiLSTM initialization

        Args:
            n_units: (int) Number of recurrent units
            rec_layers: (int) Number of recurrent layers to stack upon each other
            sigmoid: (bool) Add Sigmoid activation layer at output
            name: (str) Module name
        '''
        super().__init__(name=name)

        ## LAYERS ##
        self.mask = Masking(mask_value=-1.)

        # Set return_sequences=True in first layers of BiLSTM stacked layers
        self.rec = [Bidirectional(LSTM(n_units, return_sequences=True)) for _ in range(rec_layers-1)]
        self.rec.append(Bidirectional(LSTM(n_units)))

        # Output layer 
        if sigmoid:
            self.out = Dense(1, activation='sigmoid')
        else:
            self.out = Layer() # Identity pass through
    
    def __call__(self, input_):
        '''Baseline forward pass'''
        # Mask input
        x = self.mask(input_)
        # Pass through stacked BiLSTM layers
        for rec_layer in self.rec:
            x = rec_layer(x)
        return self.out(x)

def baseline_model(n_units=32, rec_layers=1):
    '''Baseline model (BiLSTM)
    
    Args:
        n_units: Number of recurrent units, default = 32
        rec_layers: Number of stacked recurrent layers, default = 1

    Returns:
        BiLSTM baseline model
    '''
    model = Sequential()
    model.add(Baseline(n_units, rec_layers))
    return model

def dense_discriminator(in_shape, units=64, n_classes=2, n_hidden=2):
    '''Construct a fully-connected discriminator network
    
    Args:
        in_shape: Input data shape (without batch dimension)
        units: Hidden layers output dimensionality, default = 64
        n_classes: Number of prediction classes, default = 2
        n_hidden: Number of hidden layers, default = 2

    Returns:
        Dense discriminator network    
    '''
    input_layer = Input(shape=in_shape)

    # Flatten output to pass to dense layers
    x = Flatten()(input_layer)
    # Fully connected layers
    for _ in range(n_hidden):
        x = Dense(units, activation='relu')(x)
    # Class prediction with softmax
    final_output = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=final_output)
    return model

# Note: Discriminator layers do not have a gradient with this solution
class BaselineCC(Model):
    '''
    Baseline Conditional Computing Module
    '''
    def __init__(self, in_shape, n_paths=2):
        '''BaselineCCModule initialization
        
        Args:
            n_paths: Number of sub-networks, default = 2
        '''
        super().__init__()
        self.n_paths = n_paths

        # Discriminator
        self.discriminator = dense_discriminator(in_shape)

        # Sub-networks
        self.paths = [Baseline(rec_layers=2) for _ in range(n_paths)]

    def __call__(self, input_, training=False):
        '''BaselineCCModule forward pass
        
        Args:
            input_: Input to feed to sub-networks
            probs: Predicted probabilities for choosing path

        Returns:
            Pass/Fail prediction from input evaluated on chosen sub-network 
        '''
        # Feed to discriminator first
        probs = self.discriminator(input_)
        # Which path has the greatest probability?
        which = tf.argmax(probs, axis=-1, output_type=tf.dtypes.int32)
        # Output taken from chosen sub-network
        out = tf.switch_case(which[0], {i : lambda: self.paths[i](input_) for i in range(self.n_paths)})
        return out

# Note: Discriminator layers do not have a gradient with this solution
def conditional_baseline(in_shape, n_paths=2, discrim=None):
    '''Conditional Baseline Model

    Args:
        in_shape: Input data shape (without batch dimension)
        n_paths: Number of sub-models, default = 2
        discrim: Optional pretrained discriminator model

    Returns:
        Conditional Baseline model
    '''
    input_layer = Input(shape=in_shape)

    # Discriminator layers
    if discrim is None:
        flatten = Flatten()(input_layer)
        d_1 = Dense(64, activation='relu')(flatten)
        d_2 = Dense(64, activation='relu')(d_1)
        # Get path probabilities
        probs = Dense(n_paths, activation='softmax')(d_2)
    else:
        # Freeze discriminator layers
        discrim.trainable = False
        probs = discrim(input_layer, training=False)

    # Which path has the greatest probability?
    which = tf.argmax(probs, axis=-1)
    # One-hot encode choice of path
    one_hot = tf.one_hot(which, depth=n_paths)

    paths = [Baseline(rec_layers=2)(input_layer) for _ in range(n_paths)]
    concat = Concatenate()(paths)

    # Take only output of chosen path
    final_output = tf.boolean_mask(concat, one_hot)

    model = Model(inputs=input_layer, outputs=final_output)
    return model