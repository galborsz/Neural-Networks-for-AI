import os
import sys
import numpy as np
from numpy import array
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import tensorflow as tf


def load_data(filename):
    # read data from csv file
    data = pd.read_csv(filename)
    df = pd.DataFrame(data)

    # select rows and columns
    df = df[:146]
    df.drop(df.columns[26:], axis=1, inplace=True)
    df.drop(df.columns[[0, 1, 2, 4, 5]], axis=1, inplace=True)

    # convert dataFrame into a numpy array
    result = df.to_numpy()

    return result


# split a univariate sequence into samples
# SOURCE: https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def main(argv):
    # define input sequence
    raw_seq = load_data("M3C.csv")
    #raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    print(raw_seq)
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)

    # define model
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=n_steps))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit(X, y, epochs=2000, verbose=0)

    # demonstrate prediction
    x_input = array([70, 80, 90])
    x_input = x_input.reshape((1, n_steps))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)

if __name__ == "__main__":
    main(sys.argv)
