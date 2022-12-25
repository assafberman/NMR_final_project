import datetime

import keras
import numpy as np
import tensorflow as tf
from auxiliary import prompt_message
from keras import Sequential
from keras.layers import InputLayer, Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
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
    # model = Sequential()
    # model.add(InputLayer(input_shape=(input_length, embedding_length, 1)))
    # model.add(Dense(units=300, activation='relu'))
    # model.add(Dropout(rate=0.2))
    # model.add(Flatten())
    # model.add(Dense(units=300, activation='relu'))
    # model.add(Dense(units=output_size, activation='sigmoid'))
    # model = compile_model(model, learning_rate=learning_rate)
    input_layer = Input(shape=(input_length, embedding_length, 1), name='Input')
    conv_layer1 = Conv2D(filters=50, kernel_size=(1, embedding_length), padding='valid', name='Convolution_1_kernel_1')(
        input_layer)
    conv_layer2 = Conv2D(filters=50, kernel_size=(2, embedding_length), padding='valid', name='Convolution_1_kernel_2')(
        input_layer)
    conv_layer3 = Conv2D(filters=50, kernel_size=(4, embedding_length), padding='valid', name='Convolution_1_kernel_4')(
        input_layer)
    conv_layer4 = Conv2D(filters=50, kernel_size=(8, embedding_length), padding='valid', name='Convolution_1_kernel_8')(
        input_layer)
    concat_layer = Concatenate(axis=1, name='Concatenation')([conv_layer1, conv_layer2, conv_layer3, conv_layer4])
    maxpool_layer = MaxPooling2D(pool_size=(2,1), name='Maxpool')(concat_layer)
    conv_layer1 = Conv2D(filters=50, kernel_size=(1, 1), padding='valid', name='Convolution_2_kernel_1')(
        maxpool_layer)
    conv_layer2 = Conv2D(filters=50, kernel_size=(2, 1), padding='valid', name='Convolution_2_kernel_2')(
        maxpool_layer)
    conv_layer3 = Conv2D(filters=50, kernel_size=(4, 1), padding='valid', name='Convolution_2_kernel_4')(
        maxpool_layer)
    conv_layer4 = Conv2D(filters=50, kernel_size=(8, 1), padding='valid', name='Convolution_2_kernel_8')(
        maxpool_layer)
    concat_layer = Concatenate(axis=1, name='Concatenation_2')(
        [conv_layer1, conv_layer2, conv_layer3, conv_layer4])
    maxpool_layer = MaxPooling2D(pool_size=(2, 1), name='Maxpool_2')(concat_layer)
    flatten_layer = Flatten(name='Flatten')(maxpool_layer)
    dropout_layer = Dropout(rate=0.2, name='Dropout')(flatten_layer)
    dense_layer = Dense(units=256, activation='relu', name='Fully_Connected')(dropout_layer)
    dropout_layer = Dropout(rate=0.2, name='Dropout_2')(dense_layer)
    dense_layer = Dense(units=512, activation='relu', name='Fully_Connected_2')(dropout_layer)
    output_layer = Dense(units=output_size, activation='sigmoid', name='Output')(dense_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer, name='NMR_Model')
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
                    save_path='./pre_trained', verbose=True, early_stopping=True, reduce_LR=True):
    log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                               restore_best_weights=True, min_delta=0.0001, verbose=1)
    reduce_LR_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                              patience=10, min_delta=0.0001, verbose=1)
    if verbose:
        prompt_message('Model initialization started.')
    new_model = initialize_model(input_length=input_size, embedding_length=embedding_size, output_size=output_size)
    if verbose:
        prompt_message('Model initialized successfully.')
    new_model.summary()
    tf.keras.utils.plot_model(new_model, to_file='{}/model_graph.png'.format(log_dir), show_shapes=True,
                              show_layer_activations=True)
    if verbose:
        prompt_message('Model fitting started.')
    callback_list = []
    if early_stopping: callback_list.append(early_stopping_callback)
    if reduce_LR: callback_list.append(reduce_LR_callback)
    callback_list.append(tensorboard_callback)
    new_model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                  callbacks=callback_list)
    if verbose:
        prompt_message('Model fitted successfully.')
    new_model.save(save_path)
    if verbose:
        prompt_message('Model saved.')
    return new_model
