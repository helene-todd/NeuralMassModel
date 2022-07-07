import matplotlib.pyplot as plt
import random as random
import math as math
import numpy as np
import os

# Fixed parameters
dt = 10**(-3)
neurons = 10**4
sel_neurons = 100
g, J = 4, -10
tau, tau_d = 1, 50
eta_mean = 1
max_time = 300
delta = 1

current_start, current_stop = 150, 151
I0 = 42

vr, vth = -40, 100
a = abs(vth/vr)

def euler(I, dt=10**(-3)):
    v, vs, r, s = [0], [0], [0], [0]
    current = I
    for i in range(1, int(max_time/dt)) :
        r.append(r[i-1] + dt*(delta/(tau*math.pi) + 2*r[i-1]*vs[i-1] - g*r[i-1])/tau)
        vs.append(vs[i-1] + dt*(vs[i-1]**2 + eta_mean - pow(math.pi*tau*r[i-1],2) + J*tau*s[i-1] + g*math.log(a)*tau*r[i-1] + current[i-1])/tau)
        v.append(vs[i-1] + tau*math.log(a)*r[i-1] )
        s.append(s[i-1] + dt*(-s[i-1] + r[i-1])/tau_d)
    return np.array(v), np.array(r)

def R(z) :
    return (1/(math.pi*tau))*((1-z.conjugate())/(1+z.conjugate())).real

def V(z) :
    return ((1-z.conjugate())/(1+z.conjugate())).imag

def dF(z, s, I):
    return (1/tau)*( ((z+1)**2) * (1j/2) * (eta_mean + J*tau*s + g*math.log(a)*R(z) + g*V(z) + I) - ((z+1)**2)*delta/2 - ((z-1)**2)*(1j/2) + (1-z**2)*g/2 )

def euler_kuramoto(current, dt=10**(-3)):
    z = [1 + 0*1j] # b/c v0 = r0 = 0
    s = [0]
    I = current
    for i in range(1, int(max_time/dt)) :
        z.append(z[i-1] + dt*dF(z[i-1], s[i-1], I[i-1]))
        s.append(s[i-1] + dt*(-s[i-1] + R(z[i-1]))/tau_d)
    z = np.array(z)
    return np.absolute(z)

# Generating subplots
fig, ax = plt.subplots(4, 1, figsize=(16,6), sharex=True)

# Initialize input current vector
steps = int((max_time)/dt)
current = [0 for s in range(steps+1)]
for i in range(len(current)):
    if i >= int(current_start/dt) and i <= int(current_stop/dt) :
        current[i] = I0

# Analytical solution
v_sol, r_sol = euler(I=current)

# Plotting injected current I
times = [float(dt*k) for k in range(len(current))]
ax[0].set_ylabel('I(t)')
ax[0].plot(times, current, color='black')
ax[0].set_xlabel('time')

# Plotting voltage average
ax[1].set_ylabel('v_avg(t)')
ax[1].plot([float(dt*k) for k in range(len(v_sol))], v_sol, c='r', label='analytical')

# Plotting firing rate average
ax[2].set_ylabel('r(t)')
ax[2].plot([float(dt*k) for k in range(len(r_sol))], r_sol, c='r')

# Plotting Kuramoto order parameter abs value
z_sol = euler_kuramoto(current)
ax[3].plot([float(dt*k) for k in range(len(z_sol))], z_sol, c='r')
ax[3].set_ylim(0, 1)
ax[3].set_ylabel('$|Z|$')
print(z_sol)

#ax[1].legend(loc='upper right')
plt.tight_layout()
plt.show()
plt.close()
