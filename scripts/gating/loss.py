import tensorflow as tf
from tensorflow.math import maximum

class LossL1Reg(tf.Module):
    def __init__(self, alpha=1):
        super(LossL1Reg, self).__init__()
        self.l = tf.keras.losses.BinaryCrossentropy()
        self.reg = tf.keras.regularizers.L1(l1=alpha)

    def __call__(self, y_true, y_pred, meta, training=True):
        return self.l(y_true, y_pred) + self.reg(meta['mask'].hard)

class LossL2Reg(tf.Module):
    def __init__(self, alpha=1):
        super(LossL2Reg, self).__init__()
        self.l = tf.keras.losses.BinaryCrossentropy()
        self.reg = tf.keras.regularizers.L2(l2=alpha)

    def __call__(self, y_true, y_pred, meta, training=True):
        return self.l(y_true, y_pred) + self.reg(meta['mask'].hard)


class CustomLoss(tf.Module):
    def __init__(self, alpha=1, n_epochs=100):
        super(CustomLoss, self).__init__()
        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        self.sparsity_loss = SparsityCriterion(n_epochs, sparsity_target=0.5)
        self.alpha = alpha

    def __call__(self, y_true, y_pred, meta, training=True):
        l = tf.cast(self.loss_object(y_true, y_pred), tf.float32)
        l += self.alpha * self.sparsity_loss(meta, training=training)
        return l

    def set_epochs(self, val):
        self.sparsity_loss.set_epochs(val)

class SparsityCriterion(tf.Module):
    '''Defines sparsity loss
    
    Parts:
        - Network loss: MSE between current and target sparsity
        - Mask loss: (only one atm) sparsity percentage for a mask must lie between upper and lower bound.

    Loss is annealed.
    '''
    
    def __init__(self, n_epochs, sparsity_target=0.5):
        super(SparsityCriterion, self).__init__()

        self.n_epochs = n_epochs
        self.sparsity_target = sparsity_target

    def __call__(self, meta, training=True):
        
        # If additional masks are added, use loop over masks to adapt
        m = meta['mask']
        percent = m.active / m.total

        # Init masks loss as tensor
        mask_loss = tf.zeros_like(percent)
        if training:
            progress = meta['epoch'] / self.n_epochs
            upper_bound = 1. - progress * (1. - self.sparsity_target)
            lower_bound = progress * self.sparsity_target
            mask_loss += maximum(0., percent - upper_bound) ** 2
            mask_loss += maximum(0., lower_bound - percent) ** 2

        # Network loss
        net_loss = (percent - self.sparsity_target) ** 2

        return net_loss + mask_loss

    def set_epochs(self, val):
        self.n_epochs = val