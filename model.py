import tensorflow as tf
from auxiliary import prompt_message


def initialize_model(input_length=100, embedding_length=38, output_size=512, learning_rate=0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_length, embedding_length, 1)))
    model.add(tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 5), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 1)))
    model.add(tf.keras.layers.Conv2D(filters=100, kernel_size=(5, 5), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 1)))
    model.add(tf.keras.layers.Conv2D(filters=200, kernel_size=(3, 3), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=(output_size*2)))
    model.add(tf.keras.layers.Dense(units=output_size))
    model.build(input_shape=(input_length, embedding_length, 1))
    compile_model(model, learning_rate=learning_rate)
    return model


def compile_model(model: tf.keras.Model, learning_rate=0.001):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.Huber(),
                  metrics=tf.keras.metrics.CosineSimilarity())


def cosine_similarity(y_pred, y_true):
    cs = tf.keras.metrics.CosineSimilarity()
    cs.update_state(y_true, y_pred)
    return cs.result().numpy()


def import_pre_trained(model_path):
    return tf.keras.models.load_model(model_path)


def train_new_model(x_train, y_train, input_size=100, embedding_size=38, output_size=512, epochs=50, batch_size=64, validation_split=0.2,
                    save_path='./pre_trained', verbose=True, early_stopping=True):
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2)
    if verbose:
        prompt_message('Model initialization started.')
    new_model = initialize_model(input_length=input_size, embedding_length=embedding_size, output_size=output_size)
    if verbose:
        prompt_message('Model initialized successfuly.')
    new_model.summary()
    if verbose:
        prompt_message('Model fitting started.')
    if early_stopping:
        new_model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                      callbacks=[early_stopping_callback])
    else:
        new_model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    if verbose:
        prompt_message('Model fitted successfuly.')
    new_model.save(save_path)
    if verbose:
        prompt_message('Model saved.')
    return new_model