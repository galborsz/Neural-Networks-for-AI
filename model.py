import os
import sys
import statistics
import numpy as np
from numpy import array
import pandas as pd
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def load_data(filename):
    # read data from csv file
    data = pd.read_csv(filename)
    df = pd.DataFrame(data)

    # select rows and columns
    df = df[:475]
    df.drop(df.columns[74:], axis=1, inplace=True)
    df.drop(df.columns[[0, 1, 2, 3, 4, 5]], axis=1, inplace=True)
 
    X = np.array(df.values)

    return X

def detrending (X):
    #Find the trend of the data by using a moving average
    all = []
    for serie in X:
        avg = []
        avg.append(np.mean([serie[0], serie[1], serie[2]]))
        for i in range(1, len(serie)-1):
            value = np.mean([serie[i-1], serie[i], serie[i+1]])
            avg.append(value)
        value = avg.pop()
        avg.append(value)
        avg.append(value)
        all.append(avg)
    
    trend = np.array(all)

    #find the detrended version of the data by substracting the trend from the original data
    detrended = X - trend
    
    return detrended, trend


# split the timeseries into input data and predictions
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

    ax.plot(list(pred[0]),yhat[0],'x', label="k=6")

    ax.set_xlabel("original value",fontsize=20)
    ax.set_ylabel("predicted value",fontsize=20)

    ax.set_xlim()
    ax.legend(prop={'size': 13})
    plt.savefig('plot.png')
    plt.show()

def create_model():
    # define model
    model = Sequential()
    model.add(Dense(1, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.add(Dense(6))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model

def main(argv):
    # define input sequence
    raw_seq = load_data("new.csv")

    # split between training and testing data
    test_dataset, training_dataset = train_test_split(raw_seq, train_size=0.8, test_size=0.2, random_state = 3)

    Test_Loss = []
    Test_Accuracy = []
    mean_per_run = []

    for x in range(5):
        kf = KFold(n_splits=10) 
        i=0
        for train, test in kf.split(training_dataset):
            i=i+1
            print("Running Fold", i,"/", 10)
            model = None # Clearing the NN.
            model = create_model()

            #split timeseries between input data and values to predict
            data, pred = split_sequence(training_dataset[train]) 

            #find trend and detrended version of the training data set
            detrended, trend = detrending(data)
            
            #input x values for polynomials
            x = range(len(trend[1]))
            x2 = range(len(trend[0]), 68)
            
            #find best adjusted polynomial to each trend, make polynomial larger and substract these forecasted values from the original prediction values
            detrended_pred = []
            for serie_trend, serie_pred in zip(trend, pred):
                trend_coeff = np.polyfit(list(x), serie_trend, deg = 3)
                polyn = np.poly1d(trend_coeff)
                value = serie_pred - polyn(x2)
                detrended_pred.append(value)

            # fit model detrended data
            model.fit(np.array(detrended), np.array(detrended_pred), batch_size=14 , epochs=30, verbose=1)
        
            #split timeseries between input data and values to predict
            data_test, pred_test = split_sequence(training_dataset[test]) 

            #evaluate the model with the training data
            results = model.evaluate(np.array(data_test), np.array(pred_test))
            print("test loss, test acc:", results)
            Test_Loss.append(results[0])
            Test_Accuracy.append(results[1])

        mean_per_run.append(np.mean(Test_Accuracy) * 100)
    
    #mean of the accuracies obtained in the 5 iterations of the evaluation process
    total_mean = np.mean(mean_per_run)
    print("Accuracy of Model with training data:", total_mean) 

    #find trend and detrended version of the testing data set
    test_detrended, test_trend = detrending(test_dataset)

    #split timeseries between input data and values to predict the detrended version of the test data
    test_data, test_pred = split_sequence(test_detrended)

    #split timeseries between input data and values to predict the trend of the test data
    test_data_trend, test_pred_trend = split_sequence(test_trend)

    #evaluate the model with the testing data
    results_test = model.evaluate(np.array(test_data), np.array(test_pred))
    print("Test loss, Test acc:", results_test)

    # Generate predictions of the detrended data
    yhat = model.predict(np.array(test_data), verbose=0)
    print("Detrended predictions: ", yhat)

    # Generate predictions of the trend
    yhat_trend = model.predict(np.array(test_data_trend), verbose=0)
    print("Trend predictions: ", yhat_trend)

    #Final predictions: sum of the predictions of the detrended data and the predictions of the trend
    print("Final predictions: ", yhat_trend + yhat)


if __name__ == "__main__":
    main(sys.argv)