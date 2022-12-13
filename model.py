import datetime

import numpy as np
import tensorflow as tf
from auxiliary import prompt_message


def initialize_model(input_length=100, embedding_length=9, output_size=512, learning_rate=0.001):
    input_layer = tf.keras.Input(shape=(input_length, embedding_length, 1))
    conv_initial = tf.keras.layers.Conv2D(filters=200, kernel_size=(3, 3), padding='same')(input_layer)
    for i in range(4):
        rcl_conv_layer = tf.keras.layers.Conv2D(filters=200, kernel_size=(3, 3), padding='same')(conv_initial)
        rcl_add_layer = tf.keras.layers.Add()([conv_initial, rcl_conv_layer])
        rcl_maxpool_layer = tf.keras.layers.MaxPooling2D(pool_size=(2,1))(rcl_add_layer)
        rcl_batch_layer = tf.keras.layers.BatchNormalization()(rcl_maxpool_layer)
        conv_initial = rcl_batch_layer
    flatten_layer = tf.keras.layers.Flatten()(rcl_batch_layer)
    dense1 = tf.keras.layers.Dense(units=(output_size*2), activation='sigmoid')(flatten_layer)
    output_layer = tf.keras.layers.Dense(units=output_size, activation='sigmoid')(dense1)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='NMR_model')
    model = compile_model(model, learning_rate=learning_rate)
    return model


def compile_model(model: tf.keras.Model, learning_rate=0.001):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.Huber(),
                  metrics=tf.keras.metrics.CosineSimilarity())
    return model


def cosine_similarity(y_pred, y_true):
    cs = tf.keras.metrics.CosineSimilarity()
    cs.update_state(y_true, y_pred)
    return cs.result().numpy()


def import_pre_trained(model_path):
    return tf.keras.models.load_model(model_path)


def train_new_model(x_train, y_train, input_size=100, embedding_size=9, output_size=512, epochs=50, batch_size=64, validation_split=0.2,
                    save_path='./pre_trained', verbose=True, early_stopping=True):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    if verbose:
        prompt_message('Model initialization started.')
    new_model = initialize_model(input_length=input_size, embedding_length=embedding_size, output_size=output_size)
    if verbose:
        prompt_message('Model initialized successfuly.')
    new_model.summary()
    #tf.keras.utils.plot_model(new_model, to_file='{}/model_graph.png'.format(log_dir), show_shapes=True, show_layer_activations=True)
    if verbose:
        prompt_message('Model fitting started.')
    if early_stopping:
        new_model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                      callbacks=[early_stopping_callback, tensorboard_callback])
    else:
        new_model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                      callbacks=[tensorboard_callback])
    if verbose:
        prompt_message('Model fitted successfuly.')
    new_model.save(save_path)
    if verbose:
        prompt_message('Model saved.')
    return new_model
