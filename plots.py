import matplotlib.pyplot as plt
import random as random
import math as math
import numpy as np
import os

# Fixed parameters
dt = 10**(-3)
neurons = 10**4
sel_neurons = 1000

def euler(I, dt=10**(-3), max_time=20, eta_mean=4, tau=1, tau_d=5, delta=1, g=0, J=10, vr=-100, vth=100):
    v, r, s = [0], [0], [0]
    current = I
    a = -vth/vr
    for i in range(1, int(max_time/dt)) :
        r.append(r[i-1] + dt*(delta/(tau*math.pi) + 2*r[i-1]*v[i-1] - g*r[i-1])/tau)
        v.append(v[i-1] + dt*(v[i-1]**2 + eta_mean - pow(math.pi*tau*r[i-1],2) + J*tau*s[i-1] + g*math.log(a)*tau*r[i-1] + current[i-1])/tau)
        s.append(s[i-1] + dt*(-s[i-1] + r[i-1])/tau_d)
    return np.array(v), np.array(r)

def R(z, eta_mean=4, tau=1, delta=1, g=0, J=10) :
    return (1/(math.pi*tau))*((1-z.conjugate())/(1+z.conjugate())).real

def V(z, eta_mean=4, tau=1, delta=1, g=0, J=10) :
    return ((1-z.conjugate())/(1+z.conjugate())).imag

def dF(z, s, I, eta_mean=4, tau=1, delta=1, g=0, J=10):
    #return (1/tau)*( ((z+1)**2) * (1j/2) * (eta_mean + J*tau*s + g*V(z) + I) - ((z+1)**2)*delta/2 - ((z-1)**2)*(1j/2) + (1-z**2)*g/2 )
    return (1/tau)*( (1j/2)*(eta_mean + J*tau*s + I)*pow(z+1, 2) - (1j/2)*pow(1-z, 2) - (delta/2)*pow(1+z, 2)  )

def euler_kuramoto(current, tau_d=5, dt=10**(-3), max_time=20):
    z = [1 + 0*1j] # b/c v0 = r0 = 0
    s = [0]
    I = current
    for i in range(1, int(max_time/dt)) :
        z.append(z[i-1] + dt*dF(z[i-1], s[i-1], I[i-1]))
        s.append(s[i-1] + dt*(-s[i-1] + R(z[i-1]))/tau_d)
    z = np.array(z)
    return np.absolute(z)

# Retrieving and processing the data
v_file = open('v_avg.dat', 'r')
v = []
for el in v_file.readline()[:-2].split(',') :
    v.append(float(el))

r_file = open('r_avg.dat', 'r')
r = []
for el in r_file.readline()[:-2].split(',') :
    r.append(float(el))

z_file = open('z.dat', 'r')
z = []
for el in z_file.readline()[:-2].split(',') :
    z.append(float(el))
z = np.array(z)

input_file = open('input_current.dat', 'r')
current = []
for el in input_file.readline()[:-2].split(',') :
    current.append(float(el))
current = np.array(current)

raster_file = open('raster.dat', 'r')
times_raster, raster = [], []
for line in raster_file :
    times_raster.append(float(line.split('  ')[0]))
    raster.append(int(line.split('  ')[1].rstrip()))

# Generating subplots
fig, ax = plt.subplots(5, 1, figsize=(10,6), sharex=True)

# Analytical solution
v_sol, r_sol = euler(I=current)

# Plotting injected current I
times = [float(dt*k) for k in range(len(current))]
ax[0].set_ylabel('I(t)')
ax[0].plot(times, current, color='black')
ax[0].set_xlabel('time')

# Plotting voltage average
times = [float(dt*k) for k in range(len(v))]
ax[1].set_ylabel('v_avg(t)')
ax[1].plot(times, v, c='k', label='numerical')
ax[1].plot([float(dt*k) for k in range(len(v_sol))], v_sol, c='r', label='analytical')

# Plotting firing rate average
times = [float(dt*k) for k in range(len(r))]
ax[2].set_ylabel('r(t)')
ax[2].plot(times, r, c='k')
ax[2].plot([float(dt*k) for k in range(len(r_sol))], r_sol, c='r')

# Plotting Kuramoto order parameter abs value
z_sol = euler_kuramoto(current)
times = [float(dt*k) for k in range(len(z))]
ax[3].plot(times, z, c='k')
ax[3].plot([float(dt*k) for k in range(len(z_sol))], z_sol, c='r')
ax[3].set_ylim(0, 1)
ax[3].set_ylabel('$|Z|$')
print(z_sol)

# Plotting raster plot for 300 neurons
ax[4].set_ylabel('neuron index')
ax[4].scatter(times_raster, raster, s=0.9, c='k')
ax[4].set_xlim(0, 20)
ax[4].set_ylim(0, sel_neurons)
ax[4].set_xlabel('time')

ax[1].legend(loc='upper right')
plt.tight_layout()
plt.savefig('full_model.png')
plt.show()
plt.close()
