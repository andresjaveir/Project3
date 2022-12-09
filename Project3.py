# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:19:34 2022

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import copy 

np.random.seed()

def init_lattice(n):  ## Create a nxn lattice with random spin configuration

   
    
    lattice = np.random.choice([1, -1], size=(n, n))
    return lattice



def deltaE(S0, Sn, J, H): ## Energy difference for a spin flip'''

    
    
    return 2*S0*(H + J*Sn)


def MC_ising(n, T, lattice, H=0, J=1): ## Metropolis Monte Carlo (FSMC)

    
    
    N = n**2

    
    for step in range(N):

        i = np.random.randint(n)    ## We select a random spin in the lattice
        j = np.random.randint(n)

        # Periodic Boundary Condition
        Sn = lattice[(i - 1) % n, j] + lattice[(i + 1) % n, j] + \
            lattice[i, (j - 1) % n] + lattice[i, (j + 1) % n]

        dE = deltaE(lattice[i, j], Sn, J, H)   ## The energetic difference 
                                                # between the old configuration
                                                # and the new one

        if dE <= 0: # or np.random.random() < np.exp(-dE/T):  ## (T = 0)
            lattice[i, j] = -lattice[i, j]    ## If the energy does not increase
                                            # the new state is accepted.
            
    M = (1/N)*np.sum(lattice)     ## We calculate the magnetization
    
    return M, lattice




n = 10            ## Number of spins per side
Nsteps = 1000     ## Number of steps for the simulation
T = 0               ## Temperature
M1 = np.zeros(Nsteps)   ## Empty array for the magnetization
lattice = init_lattice(n)  ## Initial lattice configuration
lattice1 = copy.deepcopy(lattice)  ## We copy the latticesfor later
lattice2 = copy.deepcopy(lattice)  ## We copy the lattice for later
M1[0] = (1/n**2)*np.sum(lattice)   ## The initial value of the magnetization

for k in range(1, Nsteps):      ## We run the simulation
    
    M1[k], lattice1 = MC_ising(n, T, lattice1, H=0, J=1)



Time = np.linspace(0, Nsteps, Nsteps)   ## The array of time




plt.figure(figsize=(8,6))    ## We plot the time
plt.title("Time-step Monte Carlo")
plt.plot(Time, np.abs(M1), color="lime")
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Magnetization")
plt.xscale('log')
plt.yscale('log')


#%%



def CTMC_ising(n, lattice, t, H=0, J=1):  ## We define the continuous time
                                            # Monte  Carlo (CTMC)   
    np.random.seed()
    N = n**2
    g = np.zeros(5)      ## 5 Different type of spins in my lattice
    P = np.array([1, 0, 1, 0, 1])*(1/N) ## The probability for each spin to flip
    lattice_aux=np.zeros((n,n))     ## An auxiliar lattice, to label each spin
    
    
    for i in range(n):
        for j in range(n):

            
            S0 = lattice[i, j]
        # Periodic Boundary Condition
            Sn = np.array([lattice[(i - 1) % n, j], lattice[(i + 1) % n, j],
                lattice[i, (j - 1) % n], lattice[i, (j + 1) % n]])
            count = np.count_nonzero(Sn == S0)
            
            if count == 0.:   ##  We label each spin judging by its type
                g[0] += 1
                lattice_aux[i,j] = 0
                
            if count == 4.:
                g[1] += 1
                lattice_aux[i,j] = 1
            
            if count == 1.:
                g[2] +=1
                lattice_aux[i,j] = 2
            
            if count == 3.:
                g[3] += 1
                lattice_aux[i,j] = 3
                
            if count == 2.:
                g[4] += 1
                lattice_aux[i,j] = 4
    
    
    

    
    Q = np.sum(g*P)    
    Q2 = 1-Q
    M = (1/N)*np.sum(lattice)
    
    if (Q2 == 1 or Q2 ==0):  ## To avoid problems with infinite values
        
        dt = 1
        t += dt

        
       
                
    
        
    
    else:

        
        h = 1-np.random.rand()
        dt = 1+int(np.log(h)/np.log(Q2))   ## We update the time
    
        t += dt
        choice1 = g*P  
        
        a = np.random.rand()*np.sum(choice1)  ## We choose a type of 
                                               # spin to be flipped
        
        if a <= choice1[0]:
            l = 0
        
        if choice1[0] < a <= choice1[0]+choice1[2]:
            l = 2
            
        if choice1[0]+choice1[2] < a:
            l=4
            
        i = np.random.randint(0,10)   ## We choose a random spin
        j = np.random.randint(0,10)
        
        while lattice_aux[i,j] != l:  ## We filter the spin chosen so it matches
                                        # the type of spin selected
            i = np.random.randint(0,10)
            j = np.random.randint(0,10) 
        
        lattice[i,j] = -lattice[i,j]

    return M, t, lattice




n = 10        ## Number of spins per side
Nsteps = 1000   ## Number of steps in my simulation
M = np.zeros(Nsteps)  ## Array for the magnetization
t = 0           ## Initial value of time
tplot=np.zeros(Nsteps)      ## Array for the time
#lattice = init_lattice(n)   ## We generate a random initial configuration, it
                              # is commented since we will use the one previously
                              # generated, so the initial configuration for 
                              # both the FSMC and the CTMC is the same

np.random.seed()

M[0] = (1/n**2)*np.sum(lattice)  ## Initial value of the magnetization

for k in range(1, Nsteps):  ## We begin the CTMC simulation
    
    M[k], t, lattice = CTMC_ising(n, lattice, t)
    
    tplot[k] = t


plt.figure(figsize=(8,6)) ## We plot the evolution of the system
plt.title("Continuous time Monte Carlo")
plt.plot(tplot, np.abs(M), color="green")
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Magnetization")
plt.xscale('log')
plt.yscale('log')


plt.figure(figsize=(8,6))  ## We plot both the evolution for the FSMC and the CTMC
plt.title("CTMC vs FSMC")
plt.plot(tplot, np.abs(M), color="green", label = "CTMC")
plt.plot(Time, np.abs(M1), color="lime", label = "FSMC")
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Magnetization")
plt.legend(loc="best")
plt.xscale('log')
plt.yscale('log')



## We plot the initial configuration

Xup = []  
Yup = []

Xdown = []
Ydown = []

for i in range(n):
    for j in range(n):
        
        if lattice2[i,j] == 1:
            Xup.append(i)
            Yup.append(j)
            
        if lattice2[i,j]== -1:
            Xdown.append(i)
            Ydown.append(j)
            

Xup = np.array(Xup)
Yup = np.array(Yup)

Xdown = np.array(Xdown)
Ydown = np.array(Ydown)

plt.figure(figsize=(8,6))
plt.title("10x10 spin lattice")
plt.plot(Xup, Yup, 'o', color = "b", marker = r'$\uparrow$', 
         markersize=17)            
plt.plot(Xdown, Ydown, 'o', color = "r", marker = r'$\downarrow$', 
         markersize=17)






#%%


## We generate a 2D square lattice of 50x50 spins and we plot it.

n = 50

lattice = init_lattice(n)


Xup = []
Yup = []

Xdown = []
Ydown = []




for i in range(n):
    for j in range(n):
        
        if lattice[i,j] == 1:
            Xup.append(i)
            Yup.append(j)
            
        if lattice[i,j]== -1:
            Xdown.append(i)
            Ydown.append(j)
            

Xup = np.array(Xup)
Yup = np.array(Yup)

Xdown = np.array(Xdown)
Ydown = np.array(Ydown)

plt.figure(figsize=(8,8))
plt.title("50x50 spin lattice")
plt.plot(Xup, Yup, 'o', color = "b", marker = r'$\uparrow$', 
         markersize=6)            
plt.plot(Xdown, Ydown, 'o', color = "r", marker = r'$\downarrow$', 
         markersize=6)


#%%

## We generate 10 different random initial 10x10 spin configurations

a = 10   ## The index for each different system.
n = 10
Nsteps = 1000
Time = np.linspace(0, Nsteps, Nsteps)
T = 0
lattice = np.zeros((n,n,a))  ## Initial arrays for each lattice.

for i in range(a):
    lattice[i] = init_lattice(n) ## We generate the 10 lattices.
    
    
M = np.zeros((a, Nsteps))  ## Array of the magnetization for each system

for i in range(a):   ## We begin the FSMC simulation for each lattice
    for j in range(Nsteps):
        M[i][j], lattice[i] = MC_ising(n, T, lattice[i])
        
        

M = np.abs(M)  ## We take the absolute value of the magnetization, as it is
                # irrelevant if the spins are up or down, we're only interested
                # in its modulus.
M_mean = np.zeros(Nsteps)

for i in range(a):  ## We now calculate the average for the 10 systems
    M_mean += M[i]/a


plt.figure(figsize=(8,6))  ## We plot it
plt.title("Mean value of M(t) for FSMC")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time")
plt.ylabel("Magnetization")
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
plt.plot(Time, M_mean, color="r")
        
        
        

plt.figure(figsize=(8,6))  ## We plot each system
plt.title("10 independent FSMC simulations")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time")
plt.ylabel("Magnetization")
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
for i in range(a):
    plt.plot(Time, M[i])




## Now, for the CTMC




a = 10
n = 10
Nsteps = 1000
lattice1 = np.zeros((n,n,a))
t = np.zeros(a)
tplot = np.zeros((a, Nsteps))

for i in range(a):  ## We generate the lattices, like before.
    lattice1[i] = init_lattice(n)
    
    
M1 = np.zeros((a, Nsteps))

for i in range(a):  ## We begin the simulation
    for j in range(Nsteps):
        M1[i][j], t[i], lattice1[i] = CTMC_ising(n, lattice1[i], t[i])
        tplot[i][j] = t[i]
        
M1 = np.abs(M1)         ## We take the modulus

plt.figure(figsize=(8,6)) ## We plot each system 
plt.title("10 independent CTMC simulations")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time")
plt.ylabel("Magnetization")
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
for i in range(a):
    plt.plot(tplot[i], M1[i])



plt.figure(figsize=(8,6))  ## We plot the 10 CTMC and the 10 FSMC to compare them
plt.title("10 independent CTMC and 10 independent FSMC")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time")
plt.ylabel("Magnetization")
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
plt.plot(tplot[0], M1[0], color = 'green', label = 'CTMC')
plt.plot(Time, M[0], color = 'lime', label = 'FSMC')
plt.legend(loc='best')
for i in range(1, a):
    plt.plot(tplot[i], M1[i], color = 'green')    
    plt.plot(Time, M[i], color='lime')



tmax = int(np.amax(tplot))
M1_sampled = np.zeros((a, tmax))  ## To get the mean value, we need to sample 
                                    # the magnetization.

t = 0
for k in range(a):
    
    for t in range(tmax): ## Our time step is 1 second.
        
            
        for j in range(1, Nsteps):
                
            if tplot[k][j]>t: ## We take the value of the last time the system
                                # evolved
                M1_sampled[k][t] = M1[k][j-1]
                break
                
            if np.all(tplot[k]<=t): ## If the system stopped evolving, we take 
                                    # the last value of the magnetization, as it
                                    # has reached equilibrium
                M1_sampled[k][t] = max(M1[k])
                break

                


    
t_sampled = np.linspace(0, tmax, len(M1_sampled[1])) ## The array of times
    
plt.figure(figsize=(8,6)) ## We plot the sampled magnetization for the 10 systems
plt.title("10 independent CTMC simulations sampled")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time")
plt.ylabel("Magnetization")
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
for i in range(a):
    plt.plot(t_sampled, M1_sampled[i])    




M1_mean = np.zeros(tmax) ## Now we get the mean value.

for i in range(a):
    
    M1_mean += M1_sampled[i]/a
    


plt.figure(figsize=(8,6)) ##  We plot the mean value
plt.title("Mean value of M(t) for CTMC")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time")
plt.ylabel("Magnetization")
plt.grid('on', linestyle='--', linewidth=1, alpha=0.7)
plt.plot(t_sampled, M1_mean, color = "red")






            