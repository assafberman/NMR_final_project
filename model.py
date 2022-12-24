import datetime

import numpy as np
import tensorflow as tf
from auxiliary import prompt_message
from keras import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.metrics import CosineSimilarity
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
import os


def initialize_model(input_length=100, embedding_length=48, output_size=512, learning_rate=0.001):
    """
    Initializing RCNN model
    :param input_length: length of 13C chemical shifts
    :param embedding_length: length of data of single peak (shift, multiplicity, intensity)
    :param output_size: output binary vector (length of Morgan fingerprint)
    :param learning_rate: learning rate of ADAM optimizer
    :return: returns model object
    """
    model = Sequential()
    model.add(InputLayer(input_shape=(input_length, embedding_length, 1)))
    model.add(Conv2D(filters=50, kernel_size=(3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.1))
    model.add(Conv2D(filters=100, kernel_size=(3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=200, activation='tanh'))
    model.add(Dense(units=output_size, activation='sigmoid'))
    model = compile_model(model, learning_rate=learning_rate)
    return model


def compile_model(model: tf.keras.Model, learning_rate=0.001):
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=MeanSquaredError(),
                  metrics=CosineSimilarity())
    return model


def cosine_similarity(y_pred, y_true):
    cs = CosineSimilarity()
    cs.update_state(y_true, y_pred)
    return cs.result().numpy()


def import_pre_trained(model_path):
    return tf.keras.models.load_model(model_path)


def train_new_model(x_train, y_train, input_size=100, embedding_size=48, output_size=512, epochs=50, batch_size=64,
                    validation_split=0.3,
                    save_path='./pre_trained', verbose=True, early_stopping=False):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_cosine_similarity', patience=10,
                                                               restore_best_weights=True, min_delta=0.001)
    log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    if verbose:
        prompt_message('Model initialization started.')
    new_model = initialize_model(input_length=input_size, embedding_length=embedding_size, output_size=output_size)
    if verbose:
        prompt_message('Model initialized successfully.')
    new_model.summary()
    tf.keras.utils.plot_model(new_model, to_file='{}/model_graph.png'.format(log_dir), show_shapes=True, show_layer_activations=True)
    if verbose:
        prompt_message('Model fitting started.')
    if early_stopping:
        new_model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                      callbacks=[early_stopping_callback, tensorboard_callback])
    else:
        new_model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                      callbacks=[tensorboard_callback])
    if verbose:
        prompt_message('Model fitted successfully.')
    new_model.save(save_path)
    if verbose:
        prompt_message('Model saved.')
    return new_model
