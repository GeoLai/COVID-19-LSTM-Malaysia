# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:00:56 2022

@author: Lai Kar Wei
"""

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

class ModDev:
    def dl_model(self, X_train, nb_class, nb_node=64,dropout_rate=0.3): 
        #values inside this is default unless stated

        model = Sequential()
        model.add(Input(shape=(np.shape(X_train)[1:]))) # LSTM,GRU,RNN accept only 3D array
        model.add(LSTM(nb_node, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(nb_node))
        model.add(Dropout(dropout_rate))
        model.add(Dense(nb_class, activation='relu'))
        model.summary()
        
        return model

class ModEval:
    def plot_hist_graph(self, hist):
        plt.figure()
        plt.plot(hist.history['mse'])
        plt.plot(hist.history['mean_absolute_percentage_error'])
        plt.legend(['training MSE', 'MAPE'])
        plt.show()