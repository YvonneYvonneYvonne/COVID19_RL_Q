import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt

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

I_0 = 1000
E_0 = 0
R_0 = 0
F_0 = 0
S_0 = N - I_0 - E_0 - R_0 - F_0

T = 180     # T is period
uprate = 0.012      # natural unemployment growth rate 

# U_0 is initial unemployment, natural rate is 3.5%，thus 35000 people
U_0 = 35000

INI = (S_0,E_0,I_0,R_0,F_0,U_0)


def funcSEIR(inivalue,_):
    Y = np.zeros(6)
    X = inivalue
    # S
    Y[0] = - (beta * X[0] * X[2]) / N
    # E
    Y[1] = (beta * X[0] * X[2]) / N - X[1] / Te
    # I
    Y[2] = X[1] / Te - gamma * X[2] - micro * X[2]
    # R
    Y[3] = gamma * X[2]
    # D
    Y[4] = micro * X[2]

    # Unemploy
    Y[5] = uprate * X[5]
    return Y

T_range = np.arange(0,T + 1)
RES = spi.odeint(funcSEIR,INI,T_range)

# The higher the penalty, the worse the performance
# basic function
# norewardSEIR=(RES[:,2]*0.5+RES[:,4]*5+RES[:,5]*0.2) / N

# for formal set function
norewardSEIR=(RES[:,2]*2+RES[:,3]*0.5+RES[:,4]*1+RES[:,5]*0.2)

# for redline function
# norewardSEIR=(RES[:,2]*2+RES[:,3]*0.5+RES[:,4]*1+RES[:,5]*0.2*(RES[:,5]/200000))

plt.plot(RES[:,0],color = 'darkblue',label = 'Susceptible',marker = '.')
plt.plot(RES[:,1]/N,color = 'orange',label = 'Exposed',marker = '.')
plt.plot(RES[:,2],color = 'red',label = 'Infection',marker = '.')
plt.plot(RES[:,3],color = 'green',label = 'Recovery',marker = '.')
plt.plot(RES[:,4],color = 'black',label = 'Death',marker = '.')
plt.plot(RES[:,5],color = 'purple',label = 'Unemployrate',marker = '.')

plt.plot(norewardSEIR,color = 'gray',label = 'norewardSEIR',marker = '.')

plt.title('Deterministic SEIR Model')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Percentage')
plt.ylim(0,1000000)
plt.show()