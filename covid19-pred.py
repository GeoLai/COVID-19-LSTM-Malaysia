# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 08:59:50 2022

@author: Lai Kar Wei
"""

#%% Importing modules
import os
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from covid19_module import ModDev
from covid19_module import ModEval

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard

#%% Constant paths
CSV_PATH_TRAIN = os.path.join(os.getcwd(), 'dataset', 'cases_malaysia_train.csv') 
CSV_PATH_TEST = os.path.join(os.getcwd(), 'dataset', 'cases_malaysia_test.csv')
LOGS_PATH = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))
MMS_PATH = os.path.join(os.getcwd(), 'model', 'mms_train.pkl') #save pickle file into model folder

#%% Step 1 Data Loading
df_train = pd.read_csv(CSV_PATH_TRAIN) # read csv paths
df_test = pd.read_csv(CSV_PATH_TEST)

#%% Step 2 Data Inspection
df_train.info() # info on the dataframe
df_train.head() # first 5 observations
df_train.describe().T # stats summary for all the features, transposing
df_train.isna().sum() # sum of NaN in train set
df_test.isna().sum() # sum of NaN in test set

df_train.isnull().sum()
df_disp = df_train[:100]

plt.figure(figsize=(20,10))
plt.plot(df_disp['date'], df_disp['cases_new'], marker='o')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('New Cases (count)')
plt.show()

#%% Step 3 Data Cleaning
df_train['cases_new'] = pd.to_numeric(df_train['cases_new'], errors='coerce')
df_train['cases_new'] = df_train['cases_new'].fillna(0)
df_train['cases_new'] = df_train['cases_new'].interpolate(method='polynomial', order=2)
df_test['cases_new'] = df_test['cases_new'].fillna(0)

df_train['cases_new'].isnull().sum()

plt.figure(figsize=(20,10))
plt.plot(df_train['date'], df_train['cases_new'], marker='o')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('New Cases (count)')
plt.show()

#%% Step 4 Features Selection
df_train['cases_new'].dtypes

X = df_train['cases_new'] # convert series to float

mms = MinMaxScaler()
X = mms.fit_transform(np.expand_dims(X, axis=-1)) # learn and transform into array

with open(MMS_PATH, 'wb') as file:
    pickle.dump(mms, file)

win_size = 30 # window size of 30 days
X_train = [] # empty list for X_train
y_train = [] # empty list for y_train

for i in range(win_size, len(X)):
    X_train.append(X[i-win_size:i])
    y_train.append(X[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

#%% Test dataset
dataset_cat = pd.concat((df_train['cases_new'], df_test['cases_new'])) #concatenate df on 'cases_new' column only due to getting last 30 days of data

length_days = len(dataset_cat) - len(df_test) - win_size
tot_input = dataset_cat[length_days:]

Xtest = mms.transform(np.expand_dims(tot_input, axis=-1))

X_test = []
y_test = []

for i in range(win_size, len(Xtest)): #this will result in list of list
    X_test.append(Xtest[i-win_size:i])
    y_test.append(Xtest[i])

X_test = np.array(X_test) # to make the list of list into array
y_test = np.array(y_test)

#%%
md = ModDev()
model = md.dl_model(X_train, nb_class=1, dropout_rate=0.2)

plot_model(model, show_shapes=True, show_layer_names=True)

#%%
model.compile(optimizer='adam', 
              loss='mse',
              metrics=['mean_absolute_percentage_error','mse'])

tensorboard_callback = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1)
#early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)#patience-let it wait for 5 times not improving on validation loss

#%%
#model training
hist = model.fit(X_train, y_train, 
                 epochs=50,
                 callbacks=[tensorboard_callback]) #early_stopping_callback])

#%%
print(hist.history.keys())

me = ModEval()
mod_eval = me.plot_hist_graph(hist)

#%%model evaluation
predicted_c19 = model.predict(X_test)

plt.figure()
plt.plot(y_test, color='red')
plt.plot(predicted_c19, color='blue')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend(['Actual', 'Predicted'])
plt.show()

#%%
actual_case = mms.inverse_transform(y_test)
predicted_case = mms.inverse_transform(predicted_c19)

plt.figure()
plt.plot(actual_case, color='red')
plt.plot(predicted_case, color='blue') #can plus/minus to minimize the offset if offset throughout is 
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend(['Actual', 'Predicted'])
plt.show()

#%%
print('MAPE: ', mean_absolute_percentage_error(actual_case, predicted_case))
print('MSE: ', mean_squared_error(actual_case, predicted_case))


