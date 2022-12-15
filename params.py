import numpy as np

m=0.0147 
a=0.16  #distance of guide string from motor
b=0.06  #length of excitation arm of motor
D=0.095  #disc diameter
d=0.048  #diameter of driving pulley
mu = 0.0001272  #friction coefficient
zeta=0.00002368  #damping ratio
omega = 3.8
k = 2.47 
I_disc = 0.0001407 
I_mass = (m*D**2)/4 
I = I_disc + I_mass 
T = 2*np.pi*np.sqrt(I/9.8)  #natural period of oscillation
D=0.095 
y0=[0,0] 
t0 = 0
t1 = 60
tspan=np.linspace(t0,t1,(t1-t0)*501) # Same frequency as the experiment (t=0.002) 