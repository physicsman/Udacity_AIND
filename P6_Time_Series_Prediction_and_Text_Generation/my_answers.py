import numpy as np

from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [series[n:n+window_size] for n in range(len(series[:-window_size]))]
    y = series[window_size:]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size, rnn_size=5):
    model = Sequential()
    model.add(LSTM(rnn_size, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text, keep_chars=[]):
    punctuation = ['!', ',', '.', ':', ';', '?']
    keep_chars += punctuation
    
    # Make sure text is lowercase and get a list of its unique characters
    # text = text.lower()
    chars = list(sorted(set(text)))
    
    # Remove characters that are not alpha or in our keep list
    rem_chars = [char for char in chars if not char.isalpha() and char not in keep_chars]
    #rem_chars = [char for char in chars if char not in keep_chars]
    
    # Create translate table for fast replacement of characters
    table = str.maketrans(''.join(rem_chars),' '*len(''.join(rem_chars)))

    return text.translate(table)

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    for i in range(0, len(text) - window_size, step_size):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars, rnn_size=200):
    model = Sequential()
    model.add(LSTM(rnn_size, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    return model

# a multi LSTM stateful hidden layer with dropout
# Reference: https://github.com/yxtay/char-rnn-text-generation/blob/master/keras_model.py
def build_part3_RNN(batch_size, window_size, num_chars, rnn_size=200, dropout=0.0):
    model = Sequential()
    model.add(LSTM(rnn_size, return_sequences=True, stateful=True,
                   dropout=dropout, batch_input_shape=(batch_size, window_size, num_chars)))
    model.add(LSTM(rnn_size, return_sequences=True, stateful=True,
                   dropout=dropout)) 
    model.add(LSTM(rnn_size, return_sequences=True, stateful=True,
                   dropout=dropout)) 
    model.add(TimeDistributed(Dense(num_chars, activation="softmax")))
    return model
