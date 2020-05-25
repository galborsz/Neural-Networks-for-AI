import numpy as numpy

#from keras.layers import LSTM, Dense, Dropout
#from keras.models import Sequential, load_model
import pandas as pd

xls = pd.read_excel("/Users/giselaalbors/desktop/M3C.xls", index_col=0)
selectedrows = xls.iloc[0:146]
selectedcolumn = selectedrows.iloc[:, 5:25]
data = selectedcolumn.values.tolist() #convert data into list
print(data)
