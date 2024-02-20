import tensorflow as tf

from tensorflow.keras.layers import Masking, Bidirectional, LSTM, Dense
from sklearn.metrics import balanced_accuracy_score

from .masking import MaskUnit
from .loss import CustomLoss


def loss_func(model, criterion, x, y, meta, training=True):
  ''' Calculate model loss

  Args:
    model: Model to evaluate loss on
    criterion: Loss criterion to calculate
    x: Model input features
    y: Input features' true targets
    meta: Meta parameters
    training: Training indicator, default = True

  Returns:
    model output, loss value  
  '''
  y_ = model(x, meta, training=training)
  return y_, criterion(y, y_, meta)


def grad(model, criterion, inputs, targets, meta=None):
  ''' Calculate backpropagation gradients

  Args:
    model: Model to optimize
    criterion: Loss criterion to use
    inputs: Input features
    targets: Input features' true targets,
    meta: Meta parameters, default = None

  Returns:
    Model output, loss value, gradients
  '''
  with tf.GradientTape() as tape:
    y_, loss_value = loss_func(model, criterion, inputs, targets, meta, training=True)
  return y_, loss_value, tape.gradient(loss_value, model.trainable_variables)


def custom_train(model, params, meta, x_train, y_train, x_val=None, y_val=None):
  '''Custom training loop

  Args:
    model: Model to train
    params: Model parameters
    meta: Meta parameters
    x_train: Training features
    y_train: Training targets
    x_val: Validation features, default = None
    y_val: Validation targets, default = None

  Returns:
    Results dictionary
  
  '''
  # Convert data to TF DataSet
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  # Convert into batched data
  train_dataset = train_dataset.batch(params['batch_size'])

  # Optimizer
  optimizer = params['optimizer']

  # Loss
  criterion = CustomLoss(n_epochs=params['epochs'])
  if 'criterion' in params:
    criterion = params['criterion']

  # Keep results for plotting
  train_loss_results = []
  train_accuracy_results = []
  val_loss_results = []
  val_accuracy_results = []

  # Metrics
  loss_avg = tf.keras.metrics.Mean()
  bal_acc = tf.keras.metrics.AUC() # Equal to balanced accuracy in binary case

  for epoch in range(params['epochs']):
    # Update meta parameters
    meta['epoch'] = epoch + 1

    # -- Training loop over batches --
    for x, y in train_dataset:
      # Optimize model
      y_, loss_value, grads = grad(model, criterion, x, y, meta)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      # Track progress
      loss_avg.update_state(loss_value)
      bal_acc.update_state(y, y_)

    # End of epoch
    e_loss = loss_avg.result()
    loss_avg.reset_state()
    train_loss_results.append(e_loss)
    e_acc = bal_acc.result()
    bal_acc.reset_state()
    train_accuracy_results.append(e_acc)
    print(f'Epoch {epoch}: \t {e_loss:.4f} loss \t&\t {e_acc:.4f} ROC AUC')

    # Validation
    if x_val is not None:
      v_pred = model(x_val, meta, training=False)
      loss_avg.update_state(criterion(y_val, v_pred, meta, training=False))
      v_loss = loss_avg.result()
      loss_avg.reset_state()
      val_loss_results.append(v_loss)
      bal_acc.update_state(y_val, v_pred)
      v_acc = bal_acc.result()
      bal_acc.reset_state()
      val_accuracy_results.append(v_acc)
      print(f'Validation {epoch}: \t {v_loss:.4f} loss \t&\t {v_acc:.4f} ROC AUC')

  # Return scores
  return {
    'train_loss': train_loss_results,
    'train_bal_acc': train_accuracy_results,
    'val_loss': val_loss_results,
    'val_bal_acc': val_accuracy_results
  }

class MaskingModel(tf.keras.Model):
  '''Custom feature masking model'''

  def __init__(self, n_groups=5, n_units=8):
    ''' Initialize custom model

    Args:
      n_groups: Number of feature groups, default = 5
      n_units: Number of recurrent units for each BiLSTM of a feature group, default = 8
    '''
    super().__init__()
    self.mask_layer = MaskUnit(n_groups=n_groups)
    self.masking = Masking(mask_value=-1.)
    self.bilstms = [Bidirectional(LSTM(n_units)) for _ in range(n_groups)]
    self.deciders = [Dense(1, activation='sigmoid') for _ in range(n_groups)]
    
  def __call__(self, x, meta, training=True):
    # Get feature mask
    mask = self.mask_layer(x, meta, training=training).hard
    
    # Time series masking
    x = self.masking(x)
    # Split x
    xs = tf.split(x, x.shape[-1], axis=-1)

    outs = [out_(bilstm(in_)) for in_, bilstm, out_ in zip(xs, self.bilstms, self.deciders)]
    outs = tf.concat(outs, axis=-1)

    return tf.reduce_sum(outs * mask, axis=-1) / tf.maximum(tf.reduce_sum(mask, axis=-1), 1.)

  def get_mask(self, x, meta):
    return self.mask_layer(x, meta, training=False).hard