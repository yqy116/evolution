# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:48:03 2019

@author: Kelvin Yu
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dz/dt
def model(z,t):
    alpha=1.1  #Prey growth rate
    beta=0.4 #Prey death rate
    sigma=0.1 #Predator growth rate
    gamma=0.5 #Predator death rate
    x = z[0]
    y = z[1]
    rate_x=alpha*x-beta*x*y
    rate_y=sigma*x*y-gamma*y
    total_rate = [rate_x,rate_y]
    return total_rate

# initial condition
z0 = [10,10]

# number of timesteps
n = 401

t = np.linspace(0,40,n)

# store solution
x = np.empty_like(t)
y = np.empty_like(t)
# record initial conditions
x[0] = z0[0]
y[0] = z0[1]

# solve ODE
for i in range(1,n):
    length = [t[i-1],t[i]]
    z = odeint(model,z0,length,args=())
    x[i] = z[1][0]
    y[i] = z[1][1]
    z0 = z[1] 

# plot results
plt.plot(t,x,'b-',label='x(t)')
plt.plot(t,y,'r--',label='y(t)')
plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()