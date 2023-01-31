'''
This code is a culmination of work done to investigate Bayesian Neural Networks 
as a means of capturing the aleatoric and epistemic uncertainty inherent and
present within a model. With BNNs, one is able to gather from the model where
and how uncertain the model is to better ones understanding of the data. This code
is used to extract raw experimental data from files collected on 12/17/2022 for training.

Author: Jack Sovich (jsovich@villanova.edu)
Contact: Amirhassan Abbasi (aabbasi@villanova.edu)
Edited: 1/31/2023
'''

#############################
### IMPORTS and FUNCTIONS ###
#############################

# Imports
import os
from os.path import exists
import math
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from scipy.integrate import solve_ivp

# Used to learn both values for mu and sigma in the BNN below
def normal_exp(params): 
    return tfp.distributions.Normal(loc=params[:,0:1], scale=tf.math.exp(params[:,1:2]))
def NLL(y, distr): 
    return -distr.log_prob(y) 
# Plotting average (mu) and the 95% CI of the model given the data provided
def make_plot_runs_avg(ax, preds, x, down_x, down_y, alpha_data=1):
    ax.scatter(down_x,down_y,color="steelblue", alpha=alpha_data,marker='.') #observerd      
    ax.plot(x, np.mean(preds,axis=0),color="black",linewidth=1.5)
    ax.plot(x, np.quantile(preds, 0.025, axis=0),color="red",linewidth=1.5,linestyle="--")
    ax.plot(x, np.quantile(preds, 0.975, axis=0),color="red",linewidth=1.5,linestyle="--")

#######################
### GENERATING DATA ###
#######################

filename = '0.4Damping_1to14Volts' # filename for scraping from Nonlinear Pendulum Data folder
time = 'Time (s) Run #1'
response = 'Angle (rad) Run #'
voltage = '2' # voltage value from 1-14 contained within each file to choose from

data = pd.read_csv(os.getcwd()+'\\Nonlinear Pendulum Data\\{filename}.csv'.format(filename=filename),
delimiter=',')
responses = data.loc[:,[response in i for i in data.columns]]
df = pd.concat([data[time], data.loc[:,[response in i for i in data.columns]]], axis=1)
df.drop(df.loc[df[response+voltage].isnull()].index.tolist(), inplace=True)

# Subset of data to investigate / range of values to drop from training (introduce aleatoric uncertainty)
n_min = 67.5
n_max = 86.3
low_range = 81
high_range = 81.6
df_sub = df[(df[time]>=n_min) & (df[time]<=n_max)]

# Normalize Data
norm_min = np.min(df_sub[response+voltage])
norm_max = np.max(df_sub[response+voltage])
df_sub[response+voltage] = (df_sub[response+voltage] - norm_min) / (norm_max-norm_min)
# "Missing Data" Drop (introducing aleatoric uncertainty)
down_sub = pd.concat([df_sub[df_sub[time]<low_range], df_sub[df_sub[time]>high_range]], axis=0)
# Seperate train/test data (if applicable)
n = len(down_sub)
train_df = down_sub[0:int(n*1)]
# test_df = down_sub[int(n*0.8):]
# Denormalize after generating training data
df_sub[response+voltage] = df_sub[response+voltage]*(norm_max-norm_min) + norm_min
down_sub[response+voltage] = down_sub[response+voltage]*(norm_max-norm_min) + norm_min

# Plotting the data pulled from Nonlinear Pendulum Data file
plt.subplot()
plt.plot(df[time], df[response+voltage])
plt.plot(df_sub[time], df_sub[response+voltage])
plt.axvline(x = min(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
plt.axvline(x = max(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
plt.xlabel(time,size=16)
plt.ylabel(response+voltage,size=16)
plt.title('Pendulum Response Caputured Using Photogate for {voltage}V'.format(voltage=voltage))
plt.legend(('Response','Equilibrium Response','Train/Test Split'), loc='upper left')
plt.grid()
plt.show()

# Plotting subset of data with missing values range assigned
plt.scatter(x=down_sub[time], y=down_sub[response+voltage], marker='.') # Plot the responses of each file
plt.axvline(x = min(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
plt.axvline(x = low_range, color = 'r', linestyle='dashed', label = 'axvline - full height')
plt.axvline(x = high_range, color = 'r', linestyle='dashed', label = 'axvline - full height')
plt.xlabel(time,size=16)
plt.ylabel(response+voltage,size=16)
plt.legend(('Equilibrium Response','Train/Test Split','Missing Data'),loc='upper left')
plt.title('Pendulum Response Missing {perc}% of Captured Data'.format(perc=str(round(100*(1-(n/len(df_sub)))))))
plt.grid()
plt.show()

##########################
###### TRAIN NEW BNN #####
##########################

# Bayesian NN with Dropout included during training and testing
inputs = Input(shape=(1,))
hidden = Dense(100,activation="relu")(inputs)
hidden = Dropout(0.1)(hidden, training=True)
hidden = Dense(100,activation="relu")(hidden)
hidden = Dropout(0.1)(hidden, training=True)
hidden = Dense(100,activation="relu")(hidden)
hidden = Dropout(0.1)(hidden, training=True)
hidden = Dense(100,activation="relu")(hidden)
hidden = Dropout(0.1)(hidden, training=True)
hidden = Dense(100,activation="relu")(hidden)
hidden = Dropout(0.1)(hidden, training=True)
hidden = Dense(100,activation="relu")(hidden)
hidden = Dropout(0.1)(hidden, training=True)
params_mc = Dense(2)(hidden)
dist_mc = tfp.layers.DistributionLambda(normal_exp, name='normal_exp')(params_mc) 
model_mc = Model(inputs=inputs, outputs=dist_mc)
model_mc.compile(Adam(learning_rate=0.0002), loss=NLL)
model_mc.summary()

# Training the model and assigning batch size and length of training
x = train_df[time]
y = train_df[response+voltage]
batch=32
epochs=20000

from time import time as t
clock = t()
history = model_mc.fit(x,y,epochs=epochs,verbose=0,batch_size=batch)

plt.plot(history.history['loss'])
plt.legend(['Negative Log Loss'])
plt.ylabel('NLL')
plt.xlabel('Epochs')
plt.ylim([-2,2])
plt.grid()
plt.show()
print('Training time taken: {} minutes'.format(np.round((t() - clock)/60,2))) # minutes

# Saving model weights after training
model_mc.save_weights('EXP_my_model_weights.h5')
with open("EXP_my_model_structure.json", "w") as outfile:
    json.dump(model_mc.to_json(), outfile)

##########################
####### RELOAD BNN #######
##########################

# Reloading model weights
with open('EXP_my_model_structure.json', 'r') as openfile:
    json_string = json.load(openfile)
from keras.models import model_from_json
model_mc = model_from_json(json_string)
model_mc.load_weights('EXP_my_model_weights.h5')

###########################
###### INTERPOLATION ######
###########################

# Running predictions in for loop with Droput in order to capture aleatoric uncertainty in data
from tqdm import tqdm
x_pred = df_sub[time]
runs = 200 # prediction iterations to provide a rough approximation for a Bayesian distribution
mc_cpd =np.zeros((runs,len(x_pred)))
for i in tqdm(range(0,runs)):
    # Denormalizing when prediction occurs
    mc_cpd[i,:] = np.reshape(model_mc.predict(x_pred),len(x_pred))*(norm_max-norm_min) + norm_min

# Plotting using the declared function above (mean and 95% confidence interval)
ax = plt.subplot()
make_plot_runs_avg(ax, mc_cpd, x=df_sub[time], down_x=down_sub[time], down_y=down_sub[response+voltage])
ax.axvline(x = min(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
ax.axvline(x = max(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
plt.xlabel(time,size=16)
plt.ylabel(response+voltage,size=16)
plt.title('Predictive Distribution for {runs} Runs Trained Over {epochs} epochs'.format(runs=str(runs),epochs=str(epochs)))
plt.legend(('Response','Mean','Lower Bound (95% CI)','Upper Bound (95% CI)','Train/Test Split'), loc='upper left')
plt.grid()
plt.show()

# Plotting the SD over time to visualize where more is more / less confident
sigma2 = np.quantile(mc_cpd, 0.975, axis=0) - np.quantile(mc_cpd, 0.025, axis=0)
plt.plot(df_sub[time], sigma2, linewidth=1.5, linestyle='--')
plt.axvline(x = low_range, color = 'r', linestyle='dashed', label = 'axvline - full height')
plt.axvline(x = high_range, color = 'r', linestyle='dashed', label = 'axvline - full height')
plt.xlabel(time,size=16)
plt.ylabel('Size of Confidence Interval',size=16)
plt.title('95% Confidence Interval Missing Data from {low}-{high} seconds'.format(low=str(low_range),high=str(high_range)))
plt.grid()
plt.show()

###########################
###### EXTRAPOLATION ######
###########################

# Subset of data for extrapolation (+ number of seconds to extrapolate)
n_min_ex = n_min
n_max_ex = n_max+1

df_sub_ex = df[(df[time]>=n_min_ex) & (df[time]<=n_max_ex)]
plt.scatter(df_sub_ex[time],df_sub_ex[response+voltage],marker='.')
plt.axvline(x = min(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
plt.axvline(x = max(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
plt.xlabel(time,size=16)
plt.ylabel(response+voltage,size=16)
plt.legend(('Response','Train/Test Split'),loc='upper left')
plt.title('Experimental Training Data with Extrapolation Region for Prediction')
plt.grid()
plt.show()

# Running predictions in for loop with dropout in order to capture aleatoric uncertainty in data
from tqdm import tqdm
x_pred = df_sub_ex[time]
runs = 200
mc_cpd =np.zeros((runs,len(x_pred)))
for i in tqdm(range(0,runs)):
    # Denormalizing when prediction occurs
    mc_cpd[i,:] = np.reshape(model_mc.predict(x_pred),len(x_pred))*(norm_max-norm_min) + norm_min

# Plotting using the declared function above
ax = plt.subplot()
make_plot_runs_avg(ax, mc_cpd, x=df_sub_ex[time], down_x=down_sub[time], down_y=down_sub[response+voltage])
ax.axvline(x = min(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
ax.axvline(x = max(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
plt.xlabel(time,size=16)
plt.ylabel(response+voltage,size=16)
plt.title('Predictive Distribution for {runs} Runs Trained Over {epochs} Epochs'.format(runs=str(runs),epochs=str(epochs)))
plt.legend(('Response','Mean','Lower Bound (95% CI)','Upper Bound (95% CI)','Train/Test Split'), loc='upper left')
plt.grid()
plt.show()

# Plotting the SD over time to visualize where more is more / less confident after training range
sigma2 = np.quantile(mc_cpd, 0.975, axis=0) - np.quantile(mc_cpd, 0.025, axis=0)
plt.plot(df_sub_ex[time], sigma2, linewidth=1.5, linestyle='--')
plt.axvline(x = low_range, color = 'r', linestyle='dashed', label = 'axvline - full height')
plt.axvline(x = high_range, color = 'r', linestyle='dashed', label = 'axvline - full height')
plt.axvline(x = min(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
plt.axvline(x = max(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
plt.xlabel(time,size=16)
plt.ylabel('Size of Confidence Interval',size=16)
plt.title('95% Confidence Interval Missing Data from {low}-{high} Seconds'.format(low=str(low_range),high=str(high_range)))
plt.legend(('Size of C.I.','Lower Bound (95% CI)','Upper Bound (95% CI)','Train/Test Split'), loc='upper left')
plt.grid()
plt.show()
