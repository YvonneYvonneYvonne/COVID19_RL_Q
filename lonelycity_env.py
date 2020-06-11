"""
Deep Q-network for a lonely city
This part contains:
Improved SEIR model and unemployment model
Both of them is affected by policies.

Policy:
Edition2.0 by cui 06/05/2020
there are eight policies:
which aims to different parts,
1.the first one is 'do nothing',which will keep all the parameters remain as natural

2.the second one is gently lock down, which is similar to what happens in most states in US
  this will set beta to 0.15 and unemployment uprate to 1.02 while set the gamma and micro remain a natural state

3.the third one is strictly lock down, which is similar to what happens in Wuhan in China
  this will set beta to 0.02 and unemployment uprate to 1.03 while set the gamma and micro remain as natural

4.increase the hospital capacity phase I, this will set beta to 0.5, gamma to 0.15 and micro to 0.01, others remain as natural, 
  since it is more like a cheat option, it can only be used after 30 days

5.the fourth is starting to test all likely infected people, which will set beta to 0.14 and others as natural
  since it is more like a cheat option, it can only be used after 60 days

6.increase the hospital capacity phase II, this will set beta to 0.4, gamma to 0.18 and micro to 0.008, others remain as natural, 
  since it is more like a cheat option, it can only be used after 90 days

7.the sixth one is gently open to promote the economy, so it will set beta to 0.7, in the same time the uprate will be 0.99, 
  since economy is better gamma slight increase to 0.11

8.the seventh one is totally wide open to promote the economy, so it will set beta to 0.8, in the same time the uprate will be 0.98, 
  since economy is better gamma slight increase to 0.12

l1:        do nothing              beta0.6  gamma0.1 micro0.02 uprate1.01
l2:        gently lockdown         beta0.15 gamma0.1 micro0.02 uprate1.02
l3:        strictly lockdown       beta0.02 gamma0.1 micro0.02 uprate1.03
l4:        hospital phase I        beta0.5  gamma0.15 micro0.014 uprate1.01----active at 30 days
l5:        test infected people    beta0.10 gamma0.1 micro0.02 uprate1.01----active at 60 days
l6:        hospital phase II       beta0.4 gamma0.18 micro0.008 uprate1.01----active at 120 days
l7:        gently open             beta0.7 gamma0.11 micro0.02 uprate0.99
l8:        totally open            beta0.8 gamma0.12 micro0.02 uprate0.98

we also set a red line for the unemployment rate , now we set 20%, the effect will be: base_effect*(unemploy_rate/20%)^2 

Edition1.0 by cui 06/02/2020
l0:        open and do nothing     beta0.6 gamma0.1 micro0.02 uprate1.012
l5:        half lockdown           beta0.4 gamma0.1 micro0.02 uprate1.015
l9:        totally lockdown        beta0.2 gamma0.1 micro0.02 uprate1.018
wise:      cheat option            beta0.2 gamma0.1 micro0.01 uprate1.010

The RL is in RL_brain.py.
"""
import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import tkinter as tk

# N is the total individuals in the city, one million
N = 1000000
# β is rate of transmission
beta = 0.6
# gamma is rate of recovery
gamma = 0.1
# μ is rate of mortality from the disease; 0.01 lead to 0.1，0.02 is 0.16，0.03 is 0.2
micro = 0.02
# Te is incubation period
Te = 14
T = 180     # T is period
uprate = 0.012      # natural unemployment growth rate 

class City(tk.Tk, object):
    def __init__(self):
        super(City, self).__init__()
        self.action_space = ['l1', 'l2', 'l3', 'l7', 'l8']
        self.n_actions = len(self.action_space)
        self.n_features = 4
        
    def resetcity(self):
        self.day=0
        I_0 = 1000     
        E_0 = 0     
        R_0 = 0     
        F_0 = 0     
        S_0 = N - I_0 - E_0 - R_0 - F_0     
        U_0 = 35000     # U_0 is initial unemployment, natural rate is 3.5%，thus 35000 people
        self.INI = [S_0,E_0,I_0,R_0,F_0,U_0]    
        self.RES1=[]
        self.RES2=[]
        self.RES3=[]
        self.RES4=[]
        self.RES5=[]
        self.RES6=[]
        self.re=[]
        # return observation
        return np.array([I_0,R_0,F_0,U_0]) / N 

    def step(self, action):
        self.day=self.day+1
        
        if action == 0:   # l1
            beta=0.6 
            gamma=0.1 
            micro=0.008 
            uprate=1.01
        elif action == 1:   # l2
            beta=0.15 
            gamma=0.1 
            micro=0.008 
            uprate=1.02
        elif action == 2:   # l3
            beta=0.02 
            gamma=0.1 
            micro=0.008 
            uprate=1.03
        elif action == 3:   # l7
            beta=0.7 
            gamma=0.11 
            micro=0.008 
            uprate=0.99
        elif action == 4:   # l8
            beta=0.8 
            gamma=0.12 
            micro=0.008 
            uprate=0.98
        '''
        beta=0.13 
        gamma=0.1 
        micro=0.008 
        uprate=1.01
        '''
        #print('T4')
        #print(self.INI)
        Y = np.zeros(6)
        X = self.INI
        # S
        Y[0] = round(- (beta * X[0] * X[2] ) / N+X[0])
        # E
        Y[1] = round((beta * X[0] * X[2]) / N - X[1] / Te+X[1])
        # I
        Y[2] = round(X[1] / Te - gamma * X[2] - micro * X[2]+X[2])
        # R
        Y[3] = round(gamma * X[2]+X[3])
        # D
        Y[4] = round(micro * X[2]+X[4])
        # Unemploy
        Y[5] = round(uprate * X[5])

        self.INI=Y

        self.RES1.append(Y[0])
        self.RES2.append(Y[1])
        self.RES3.append(Y[2])
        self.RES4.append(Y[3])
        self.RES5.append(Y[4])
        self.RES6.append(Y[5])

        #The higher the penalty, the worse the performance
        #base penalty function
        #self.norewardSEIR=(Y[2]*0.5+Y[4]*5+Y[5]*0.2) / N
        
        #for formal set function
        #self.norewardSEIR=(Y[2]*2+Y[3]*0.5+Y[4]*1+Y[5]*0.2) / N

        #for redline function
        #improved penalty model, set unemployment rate 20% as a red line
        self.norewardSEIR=(Y[2]*2+Y[3]*0.5+Y[4]*1+Y[5]*0.2*(Y[5]/200000)) / N

        self.re.append(self.norewardSEIR*N)
        reward = 1-self.norewardSEIR
        if reward<0:
          reward=0
        #print('T5')
        # reward function
        if self.day<360:
            done = False
        else:
            done = True
        #print (self.day)
        #print (reward)
        #print('T6')
        s_ = np.array([Y[2],Y[3],Y[4],Y[5]]) / N         # next state
        return s_, reward, done

    def draw(self):
        plt.plot(self.RES1,color = 'darkblue',label = 'Susceptible',marker = '.')
        #plt.plot(self.RES2,color = 'orange',label = 'Exposed',marker = '.')
        plt.plot(self.RES3,color = 'red',label = 'Infection',marker = '.')
        plt.plot(self.RES4,color = 'green',label = 'Recovery',marker = '.')
        plt.plot(self.RES5,color = 'black',label = 'Death',marker = '.')
        plt.plot(self.RES6,color = 'purple',label = 'Unemployrate',marker = '.')

        plt.plot(self.re,color = 'gray',label = 'norewardSEIR',marker = '.')
            
        plt.title('lonelycity')
        plt.legend()
        plt.xlabel('Day')
        plt.ylim(0,1000000)
        plt.ylabel('Number')
        plt.show()
        

