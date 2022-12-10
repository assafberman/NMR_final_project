import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from auxiliary import prompt_message, input_embedding
from database_carbon import import_db_from_pickle
from model import cosine_similarity, import_pre_trained, train_new_model

pd.set_option('display.max_columns', None)

nmr_df = import_db_from_pickle()
input_list = [x for x in nmr_df['Input']]
input_list = [input_embedding(x) for x in input_list]
output_list = [x for x in nmr_df['Morgan']]
train_test_cutoff = int(0.8 * len(nmr_df['Input'].values))
input_train = np.array(input_list[:train_test_cutoff])
input_train = np.expand_dims(input_train, axis=3)
output_train = np.array(output_list[:train_test_cutoff], dtype=int)
input_test = np.array(input_list[train_test_cutoff:])
input_test = np.expand_dims(input_test, axis=3)
output_test = np.array(output_list[train_test_cutoff:], dtype=int)
prompt_message('Database imported successfuly.')

if os.path.exists('./pre_trained'):
    user_load_model = ''
    while user_load_model != 'y' or user_load_model != 'n':
        user_load_model = input('Model exists. Load model [y/n]?')
        if user_load_model == 'y':
            model = import_pre_trained('./pre_trained')
            history = np.load('./pre_trained/history.npy', allow_pickle=True).item()
            model.summary()
            tf.keras.utils.plot_model(model, 'model_graph.png', show_shapes=True, show_layer_activations=True)
            prompt_message('Model imported successfuly.')
            break
        elif user_load_model == 'n':
            model = train_new_model(x_train=input_train, y_train=output_train)
            np.save('./pre_trained/history.npy', model.history)
            history = model.history.history
            break
else:
    prompt_message('Model wasn\'t found.')
    model = train_new_model(x_train=input_train, y_train=output_train)
    np.save('./pre_trained/history.npy', model.history)
    history = model.history.history

prompt_message('Model prediction started.')
predicted_output = model.predict(input_test)
prompt_message('Model prediction ended.')

print('Cosine similarity between predictions and label:', cosine_similarity(predicted_output, output_test))

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.plot(history['cosine_similarity'])
plt.plot(history['val_cosine_similarity'])
plt.legend(['Cosine Similarity', 'Validation Cosine Similarity'])
plt.show()
