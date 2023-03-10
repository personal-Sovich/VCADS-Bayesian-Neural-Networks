'''
This code is a culmination of work done to investigate Bayesian Neural Networks 
as a means of capturing the aleatoric and epistemic uncertainty inherent and
present within a model. With BNNs, one is able to gather from the model where
and how uncertain the model is to better ones understanding of the data. This
code will highlight the aleatoric uncertainty specifically. The data used for 
modeling is synthetically created using a differential equation solver.

Author: Jack Sovich (jsovich@villanova.edu)
Contact: Amirhassan Abbasi (aabbasi@villanova.edu)
Edited: 1/31/2023
'''

#############################
### IMPORTS and FUNCTIONS ###
#############################
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

# Sythetic data differential equation solver
def phy_func(t,y):
    return [y[1],(-(k*d**2)/(2*I))*y[0] - (zeta/I)*y[1] - (mu*np.sign(y[1]))/I  + ((k*d)/(2*I))*(np.sqrt(a**2+b**2-2*a*b*np.cos(omega*t))-(a-b))-(m*9.81*D)/(2*I)*np.sin(y[0])]
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

##################
### PARAMETERS ###
##################

m=0.0147
a=0.16  #distance of guide string from motor
b=0.06  #length of excitation arm of motor
D=0.095  #disc diameter
d=0.048  #diameter of driving pulley
mu = 0.0001272  #friction coefficient
zeta=0.00002368  #damping ratio
omega = 3.8 # Tweak for chaotic vs non-chaotic response (omega=3.8 "non-chaotic")
k = 2.47 
I_disc = 0.0001407 
I_mass = (m*D**2)/4 
I = I_disc + I_mass 
T = 2*np.pi*np.sqrt(I/9.8)  #natural period of oscillation
D=0.095 
y0=[0,0] 
t0 = 0
t1 = 60
tspan=np.linspace(t0,t1,(t1-t0)*501) # Same frequency as the experimental data (t=0.002) 

# Estabishing time range, downsampling rate, and frequency
time_stop = 60 # Time range for synethetic data creation
freq = 8 # Frequency of downsampling (necessary to avoid computational errors -- only every nth sample taken)
time = 'Time (s) Run #1'
response = 'Angle, Ch P2 (rad) Run #1'
randomizer = np.linspace(omega-0.001,omega+0.001,5)
df = pd.DataFrame()
T = pd.DataFrame()
Y = pd.DataFrame()

#######################
### GENERATING DATA ###
#######################

# For loop to solve ODE for varying frequencies
for i in range(len(randomizer)):
  omega = randomizer[i]
  solution = solve_ivp(phy_func, [t0, time_stop], y0, t_eval=tspan)
  T[time] = solution.t
  Y[response] = solution.y[1]

  data = pd.concat([T,Y], axis=1)
  down = data[0::freq]
  plt.scatter(x=down[time], y=down[response], marker='.') # Plot the responses of each file
  df = pd.concat([df,down], axis=0)
plt.xlabel(time,size=16)
plt.ylabel(response,size=16)
plt.title('Synthetic Data Derived from Dynamic System Modeling Equation for Omega = {omega}'.format(omega=str(round(omega,2))))
plt.legend(('Freq 1','Freq 2','Freq 3','Freq 4', 'Freq 5'), loc='upper left')
plt.grid()
plt.show()

# Adding noise to the data and plotting frequencies together as "realistic" training set
df.sort_values(time, inplace=True)
df[response] = df[response] + np.random.normal(0,0.15*np.abs(df[response]),len(df))
plt.scatter(df[time], df[response], marker='.')
plt.xlabel(time,size=16)
plt.ylabel(response,size=16)
plt.title('Synthetic Data for 5 Frequencies with Noise Added')
plt.grid()
plt.show()

# Subset of data to investigate / range of values to drop from training (introduce aleatoric uncertainty)
n_min = 20 
n_max = 30 
low_range = 26.3 
high_range = 26.8 
df_sub = df[(df[time]>=n_min) & (df[time]<=n_max)]

# Normalize Data
norm_min = np.min(df_sub[response])
norm_max = np.max(df_sub[response])
df_sub[response] = (df_sub[response] - norm_min) / (norm_max-norm_min)
# "Missing Data" Drop (introducing aleatoric uncertainty)
down_sub = pd.concat([df_sub[df_sub[time]<low_range], df_sub[df_sub[time]>high_range]], axis=0)
# Seperate train/test data (if applicable)
n = len(down_sub)
train_df = down_sub[0:int(n*1)]
# test_df = down_sub[int(n*0.8):]
# Denormalize after generating training data
df_sub[response] = df_sub[response]*(norm_max-norm_min) + norm_min
down_sub[response] = down_sub[response]*(norm_max-norm_min) + norm_min

# Plotting subset of data with missing values range
plt.scatter(x=down_sub[time], y=down_sub[response], marker='.') # Plot the responses of each file
plt.axvline(x = min(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
plt.axvline(x = max(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
plt.xlabel(time,size=16)
plt.ylabel(response,size=16)
plt.legend(('Response','Train/Test Split'),loc='upper left')
plt.title('Realistic Training Data Missing {perc}% of Synthetic Data'.format(perc=str(round(100*(1-(n/len(df_sub)))))))
plt.grid()
plt.show()

##########################
###### TRAIN NEW BNN #####
##########################

# Bayesian NN with dropout included during training and testing
inputs = Input(shape=(1,))
hidden = Dense(200,activation="relu")(inputs)
hidden = Dropout(0.1)(hidden, training=True)
hidden = Dense(500,activation="relu")(hidden)
hidden = Dropout(0.1)(hidden, training=True)
hidden = Dense(500,activation="relu")(hidden)
hidden = Dropout(0.1)(hidden, training=True)
hidden = Dense(500,activation="relu")(hidden)
hidden = Dropout(0.1)(hidden, training=True)
hidden = Dense(200,activation="relu")(hidden)
hidden = Dropout(0.1)(hidden, training=True)
params_mc = Dense(2)(hidden)
dist_mc = tfp.layers.DistributionLambda(normal_exp, name='normal_exp')(params_mc) 
model_mc = Model(inputs=inputs, outputs=dist_mc)
model_mc.compile(Adam(learning_rate=0.0002), loss=NLL) # Adam optimizer
model_mc.summary()

# Training the model and assigning batch size and length of training
x = train_df[time]
y = train_df[response] # normalized response for training weights of network
batch=32 # size of mini batches
epochs=20000 # length of training

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
model_mc.save_weights('SYN_my_model_weights.h5')
with open("SYN_my_model_structure.json", "w") as outfile:
    json.dump(model_mc.to_json(), outfile)

##########################
####### RELOAD BNN #######
##########################

# Reloading model weights
with open('SYN_my_model_structure.json', 'r') as openfile:
    json_string = json.load(openfile)
from keras.models import model_from_json
model_mc = model_from_json(json_string)
model_mc.load_weights('SYN_my_model_weights.h5')

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
make_plot_runs_avg(ax, mc_cpd, x=df_sub[time], down_x=down_sub[time], down_y=down_sub[response])
ax.axvline(x = min(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
ax.axvline(x = max(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
plt.xlabel(time,size=16)
plt.ylabel(response,size=16)
plt.title('Predictive Distribution for {runs} Runs Trained Over {epochs} Epochs'.format(runs=str(runs),epochs=str(epochs)))
plt.legend(('Response','Mean','Lower Bound (95% CI)','Upper Bound (95% CI)','Train/Test Split'), loc='upper left')
plt.grid()
plt.show()

# Plotting the SD over time to visualize where model is more / less confident due to aleatoric uncertainty
sigma2 = np.quantile(mc_cpd, 0.975, axis=0) - np.quantile(mc_cpd, 0.025, axis=0)
plt.plot(df_sub[time], sigma2, linewidth=1.5, linestyle='--')
plt.axvline(x = low_range, color = 'r', linestyle='dashed', label = 'axvline - full height')
plt.axvline(x = high_range, color = 'r', linestyle='dashed', label = 'axvline - full height')
plt.xlabel(time,size=16)
plt.ylabel('Size of Confidence Interval',size=16)
plt.title('95% Confidence Interval Missing Data from {low}-{high} Seconds'.format(low=str(low_range),high=str(high_range)))
plt.grid()
plt.show()

###########################
###### EXTRAPOLATION ######
###########################

# Subset of data for extrapolation (+ number of seconds to extrapolate)
n_min_ex = n_min
n_max_ex = n_max+1

df_sub_ex = df[(df[time]>=n_min_ex) & (df[time]<=n_max_ex)]
plt.scatter(df_sub_ex[time],df_sub_ex[response],marker='.')
plt.axvline(x = min(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
plt.axvline(x = max(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
plt.xlabel(time,size=16)
plt.ylabel(response,size=16)
plt.legend(('Response','Train/Test Split'),loc='upper left')
plt.title('Synthetic Training Data with Extrapolation Region for Prediction')
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
make_plot_runs_avg(ax, mc_cpd, x=df_sub_ex[time], down_x=down_sub[time], down_y=down_sub[response])
ax.axvline(x = min(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
ax.axvline(x = max(train_df[time]), color = 'g', linestyle='dashed', label = 'axvline - full height')
plt.xlabel(time,size=16)
plt.ylabel(response,size=16)
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
