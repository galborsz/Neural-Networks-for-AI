import os
import sys
import numpy as np
from numpy import array
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose


def load_data(filename):
    # read data from csv file
    data = pd.read_csv(filename)
    
    df = pd.DataFrame(data)
    # select rows and columns
    df = df[:475]
    df.drop(df.columns[74:], axis=1, inplace=True)
    df.drop(df.columns[[0, 1, 2, 3, 4, 5]], axis=1, inplace=True)
 
    #statistical test to check if our time series are non-stationary
    """X = df.values
    result = adfuller(X[0])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))"""
    X = df.values
    diff = []
    for i in range(1, len(X)):
        value = X[i] - X[i - 1]
        diff.append(value)
    

    #convert dataFrame into a numpy array
    detrended = np.array(diff)
    """result = seasonal_decompose(np.transpose(X[50]), model='additive', freq = 1)

    result.plot()
    plt.show()"""

    return detrended


# split a univariate sequence into samples
# SOURCE: https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/
def split_sequence(seq):
    X=[]
    y=[]
    for elem in seq:
        X.append(elem[:62])
        y.append(elem[62:])
    return X,y

def plot(pred, yhat):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes()

    #print(pred[0])
    #print(yhat[0])
    ax.plot(list(pred[0]),yhat[0],'x', label="k=4")

    ax.set_xlabel("original value",fontsize=20)
    ax.set_ylabel("predicted value",fontsize=20)

    ax.set_xlim()
    ax.legend(prop={'size': 13})
    plt.savefig('plot.png')
    plt.show()

def main(argv):
    #read_file = pd.read_excel (r'/Users/giselaalbors/desktop/M3C.xls', sheet_name='M3Month')
    #read_file.to_csv (r'/Users/giselaalbors/desktop/new.csv', index = None, header=True)

    # define input sequence
    raw_seq = load_data("new.csv")

    # split between training and testing data
    test_dataset, training_dataset = train_test_split(raw_seq, train_size=0.8, test_size=0.2, random_state = 3)

    data,pred=split_sequence(training_dataset)

    data=np.array(data)
    pred=np.array(pred)

    # define model
    model = Sequential()
    model.add(Dense(80, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(70, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(6))
    model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

    # fit model
    model.fit(data, pred, batch_size=14 , epochs=30, verbose=1)

    # demonstrate prediction
    x_input = np.array(data[0])
    x_input = x_input.reshape((1, 62))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)

    plot(pred, yhat)

if __name__ == "__main__":
    main(sys.argv)