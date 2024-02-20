import tensorflow as tf

import os.path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

CHECK_DIR = '../models/checkpoints/'

def train_model(model, x_train, y_train, x_val, y_val, params, *callbacks):
    '''Train model

    Args:
        model: Model to train
        x_train: Training features
        y_train: Training targets
        x_val: Validation features
        y_val: Validation targets
        params: Model parameters

    Returns:
        Training history
    '''

    # Compile the model
    model.compile(optimizer=params['optimizer'], loss=params['loss'], metrics=params['metrics'])

    # Define checkpoints filepath and create it if it does not exist
    checkpoint_filepath = CHECK_DIR + params['name'] + '/'
    if not os.path.exists(checkpoint_filepath):
        os.mkdir(checkpoint_filepath)

    # Create checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        verbose=0,
        save_best_only=True,
        save_weights_only=True)

    callback_list = [checkpoint_callback]
    for cb in callbacks:
        callback_list.append(cb)

    # Start training process
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), 
                        epochs=params['epochs'], batch_size=params['batch_size'], 
                        verbose=params['verbose'], callbacks=callback_list)
    
    return history

def eval_model(model, model_params, x_test, y_test):
    '''Evaluate model
    
    Args:
        model: Model to evaluate
        model_params: Parameters of the model to evaluate
        x_test: Test features
        y_test: Test targets
    
    Returns:
        Metric scores dictionary
    '''

    # Initialize scores dict
    scores = {}

    # Load best trained model
    load_from_best_checkpoint(model, model_params)

    # Predict labels on test data
    y_pred = model.predict(x_test)

    # Calculate metrics
    acc = tf.keras.metrics.BinaryAccuracy()
    acc.update_state(y_test, y_pred)
    auc = tf.keras.metrics.AUC()
    auc.update_state(y_test, y_pred)

    scores['acc'] = acc.result().numpy()
    scores['bac'] = balanced_accuracy_score(y_test, y_pred >= 0.5)
    scores['roc_auc'] = auc.result().numpy()

    # Report accuracy scores
    print(f'Accuracy: {scores["acc"]:.3f}')
    print(f'Balanced Accuracy: {scores["bac"]:.3f}')
    print(f'ROC AUC: {scores["roc_auc"]:.3f}')

    # Show classification report
    target_names = ['Pass', 'Fail']
    print(classification_report(y_test, y_pred >= 0.5, target_names=target_names))

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred >= 0.5, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot()
    plt.show()

    return scores

def train_eval_model(model, model_params, x_train, y_train, x_val, y_val, x_test, y_test, *callbacks):
    '''Train and evaluate model
    
    Args:
        model: Model to train and evaluate
        model_params: Model parameters
        x_train: Training features
        y_train: Training target labels
        x_val: Validation features
        y_val: Validation target labels
        x_test: Testing features
        y_test: Testing target labels

    Returns:
        (void)
    '''
    tf.random.set_seed(6) # Set seed for reproducible results

    print('#### Training model ####\n')
    train_model(model, x_train, y_train, x_val, y_val, model_params, *callbacks)
    print('\n#### Evaluating model ####\n')
    return eval_model(model, model_params, x_test, y_test)

def print_metrics(model, x_test, y_test):
    '''Print ROCAUC, BalancedAccuracy, Accuracy metrics
    
    Args:
        model: Model
        x_test: Features
        y_test: Labels
    '''
    y_pred = model(x_test, training=False)

    # Test AUC
    auc = tf.keras.metrics.AUC()
    auc.update_state(y_test, y_pred)
    print(f'AUC: {auc.result()}')

    # Test balanced accuracy
    print(f'Balanced accuracy: {balanced_accuracy_score(y_test, y_pred >= 0.5)}')

    # Test accuracy
    acc = tf.keras.metrics.BinaryAccuracy()
    acc.update_state(y_test, y_pred)
    print(f'Accuracy: {acc.result()}')

def get_metrics(model, x_test, y_test):
    '''Get ROCAUC, BalancedAccuracy, Accuracy metrics
    
    Args:
        model: Model
        x_test: Features
        y_test: Labels
    '''
    y_pred = model(x_test, training=False)

    # Test AUC
    auc = tf.keras.metrics.AUC()
    auc.update_state(y_test, y_pred)

    # Test balanced accuracy
    bal_acc = balanced_accuracy_score(y_test, y_pred >= 0.5)

    # Test accuracy
    acc = tf.keras.metrics.BinaryAccuracy()
    acc.update_state(y_test, y_pred)
    
    return auc.result().numpy(), bal_acc, acc.result().numpy()

def load_from_best_checkpoint(model, params):
    ''' Loads the best model weights into the model
    
    Args:
        model: Model to load weights for
        params: The model parameters
    '''
    assert 'name' in params

    checkpoint_filepath = CHECK_DIR + params['name'] + '/'
    model.load_weights(checkpoint_filepath).expect_partial()    