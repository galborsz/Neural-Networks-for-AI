import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

def load_data(filename):

    data = pd.read_csv(filename)
    df = pd.DataFrame(data) 
    df = df[:146]
    df.drop(df.columns[26:], axis=1, inplace=True)
    df.drop(df.columns[[0, 1, 2, 4, 5]], axis = 1, inplace = True) 
    


def main(argv):
    load_data("M3C.csv")

if __name__ == "__main__":
    main(sys.argv)