from tqdm import tqdm
import matplotlib.pyplot as plt
import random as random
import math as math
import numpy as np
import os

np.random.seed()

if not os.path.exists('v_avg.dat'):
    os.mknod('v_avg.dat')

if not os.path.exists('r_avg.dat'):
    os.mknod('r_avg.dat')

if not os.path.exists('input_current.dat'):
    os.mknod('input_current.dat')

if not os.path.exists('raster.dat'):
    os.mknod('raster.dat')

# Open files
v_file = open('v_avg.dat', 'w')
r_file = open('r_avg.dat', 'w')
s_file = open('s.dat', 'w')
z_file = open('z.dat', 'w')
input_file = open('input_current.dat', 'w')
raster_file = open('raster.dat', 'w')

# Physical parameters
neurons = 10**4
sel_neurons = 10**3
v0 = 0
vr, vp = -40, 100
a = abs(vp/vr)

g, J = 4, -2
eta_mean = 1
delta = 1
current_start, current_stop = 20, 25
I = 20

# Time parameters
t_final, t_init = 100, 0
h = 10**(-3)
tau, tau_d = 1, 10
tau_s = tau*10**(-2) # Or tau*10**(-2)
steps = int((t_final - t_init)/h)
refract_steps = int(tau/(vp*h))

# Initialize eta vector
eta = [0 for n in range(neurons)]
for n in range(neurons):
    eta[n] = delta*math.tan(math.pi*(random.random()-0.5))+eta_mean

# Initialize input current vector
current = [0 for s in range(steps+1)]
for i in range(len(current)):
    if i >= int(current_start/h) and i <= int(current_stop/h) :
        current[i] = I

# Initialize mean membrane potential vector
v_avg = [0 for s in range(steps+1)]
v_avg[0] = v0

# Initialize membrane potential matrix
# NEED TO CHANGE THIS HERE
v = [[0 for n in range(neurons)], [0 for n in range(neurons)]]
for n in range(neurons):
    v[0][n] = v0

# Initialize kuramoto order parameter vector
z = [1 for z in range(steps+1)]

# Initialize spike, synaptic activation and synaptic gating vectors
spike_times = [0 for n in range(neurons)]
fire_rate = np.array([0 for r in range(steps+1)])
syn_act = [0 for s in range(steps+1)]
s = [0 for s in range(steps+1)]

# Write first lines
v_file.write(f'{v_avg[0]}, ')
r_file.write(f'{fire_rate[0]}, ')
raster_file = open('raster.dat', 'w')
z_file.write(f'{z[0]}, ')

# Loop
for i in tqdm(range(steps+1)):

    s[i] = s[i-1] + h*(-s[i-1] + syn_act[i-1])/tau_d
    for n in range(neurons):
        if(spike_times[n] == 0 and v[0][n] >= vp):
            spike_times[n] = i
        elif (spike_times[n] == 0):
            v[1][n] = v[0][n] + h*(pow(v[0][n], 2) + eta[n] + g*(v_avg[i-1]/neurons-v[0][n]) + J*tau*s[i-1] + current[i-1])/tau
        elif (i < spike_times[n] + 2*refract_steps):
            if (i >= spike_times[n] + refract_steps):
                v[1][n] = vr
                fire_rate[i] += 1
                if (n < sel_neurons and i == spike_times[n] + refract_steps + 1):
                    raster_file.write(f'{i*h}  {n+1}\n')
                if (i < spike_times[n] + refract_steps + int(tau_s/h)):
                    syn_act[i] += 1/(tau_s*neurons)
            else:
                v[1][n] = vp
        elif (i > spike_times[n] + 2*refract_steps):
            spike_times[n] = 0

        v[0][n] = v[1][n]
        v_avg[i] += v[1][n]

    w = math.pi*tau*round(fire_rate[i]/100, 5) + 1j*(round(v_avg[i]/neurons, 5) + math.log(a)*tau*round(fire_rate[i]/100, 5))
    z[i] = abs((1-w.conjugate())/(1+w.conjugate()))

    # Save values on files
    v_file.write(f'{round(v_avg[i]/neurons, 5)}, ')
    r_file.write(f'{round(fire_rate[i]/100, 5)}, ')
    input_file.write(f'{current[i]}, ')
    z_file.write(f'{z[i]}, ')


# Close files
v_file.close()
r_file.close()
input_file.close()
raster_file.close()
