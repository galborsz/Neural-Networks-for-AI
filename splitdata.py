import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filename):
    # read data from csv file
    data = pd.read_csv(filename)
    
    df = pd.DataFrame(data)

    # select rows and columns
    df = df[:146]
    df.drop(df.columns[26:], axis=1, inplace=True)
    df.drop(df.columns[[0, 1, 2, 3, 4, 5]], axis=1, inplace=True)
 
def main(argv):
    # define input sequence
    raw_seq = load_data("M3C.csv")
    test_dataset, training_dataset = train_test_split(raw_seq, train_size=5, test_size=5)

    


if __name__ == "__main__":
    main(sys.argv)