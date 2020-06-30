import os
import sys
import numpy as np
from numpy import array
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt


def load_data(filename):
    # read data from csv file
    data = pd.read_csv(filename)
    df = pd.DataFrame(data)

    # select rows and columns
    df = df[:146]
    df.drop(df.columns[26:], axis=1, inplace=True)
    df.drop(df.columns[[0, 1, 2, 3, 4, 5]], axis=1, inplace=True)

    # convert dataFrame into a numpy array
    result = df.to_numpy()

    return result


# split a univariate sequence into samples
# SOURCE: https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/
def split_sequence(seq):
    X=[]
    y=[]
    for elem in seq:
        X.append(elem[:14])
        y.append(elem[14:])
    return X,y

def plot(pred, yhat):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes()

    print(pred[0])
    print(yhat[0])
    ax.plot(list(pred[0]),yhat[0],'x', label="k=4")

    ax.set_xlabel("original value",fontsize=20)
    ax.set_ylabel("predicted value",fontsize=20)

    ax.set_xlim()
    ax.legend(prop={'size': 13})
    plt.savefig('plot.png')
    plt.show()

def main(argv):
    # define input sequence
    raw_seq = load_data("M3C.csv")

    data,pred=split_sequence(raw_seq)

    data=np.array(data)
    pred=np.array(pred)

    # define model
    model = Sequential()
    model.add(Dense(100, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

    # fit model
    model.fit(data, pred, batch_size=14 , epochs=30, verbose=1)

    # demonstrate prediction
    x_input = np.array([940.66, 1084.86, 1244.98, 1445.02, 1683.17, 2038.15, 2342.52,2602.45, 2927.87, 3103.96, 3360.27, 3807.63, 4387.88, 4936.99])
    x_input = x_input.reshape((1, 14))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)

    plot(pred, yhat)

if __name__ == "__main__":
    main(sys.argv)
