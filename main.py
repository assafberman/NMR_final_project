import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from auxiliary import prompt_message, input_embedding, extract_maximum_multiplicity_intensity
from database_carbon import import_db_from_pickle, create_input_output_list
from model import cosine_similarity, import_pre_trained, train_new_model

pd.set_option('display.max_columns', None)

nmr_df = import_db_from_pickle()
input_list, output_list = create_input_output_list(nmr_df)
input_train, input_test, output_train, output_test = train_test_split(input_list, output_list,
                                                                      train_size=0.8, random_state=42)
input_train = np.array(input_train)
output_train = np.array(output_train, dtype=int)
input_test = np.array(input_test)
output_test = np.array(output_test, dtype=int)
print(f'input shape:{input_train.shape}\noutput shape:{output_train.shape}')
prompt_message('Database imported successfully.')

if os.path.exists('./pre_trained'):
    user_load_model = ''
    while user_load_model != 'y' or user_load_model != 'n':
        user_load_model = input('Model exists. Load model [y/n]?')
        if user_load_model == 'y':
            model = import_pre_trained('./pre_trained')
            history = np.load('./pre_trained/history.npy', allow_pickle=True).item()
            model.summary()
            tf.keras.utils.plot_model(model, 'model_graph.png', show_shapes=True, show_layer_activations=True)
            prompt_message('Model imported successfully.')
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
