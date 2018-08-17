'''
Create Dicitonaries for each channel from all channels obtained from read device
'''

import sys
import math
import numpy as np
import tables as tb
import pandas as pd
#from mp_functions import *
import threading
import time

from utils import file_create
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from k_svd_object import ApproximateKSVD

from nptdms import TdmsFile    



if len(sys.argv) == 8:
    y_file = sys.argv[1]
    f_name = sys.argv[2]
    sparsity = int(sys.argv[3])
    training_sets = int(sys.argv[4])
    dict_component = int(sys.argv[5])
    m = int(sys.argv[6])
    k = int(sys.argv[7])
else:
    #print "Please give arguments with the form of:  [MP_to_use] [y_file] [chosen_column] [Phi_file] [output_file] [#mp_sparsity]  [#training_sets]"
    print "Please give arguments with the form of:  [y_file] [output file (/without.h5)] [#mp_sparsity]  [#training_sets] [# nonzero element in dictionary] [#signal_sample] [#features(columns of y)]"
    sys.exit(0)



file = TdmsFile(y_file)
channel_objects = file.group_channels("Vibration")
print channel_objects
vibration_objects = [None] * (len(channel_objects)-1)
for i in range(0, len(channel_objects)-1):
	vibration_objects[i] = channel_objects[i].data
#print np.shape(vibration_objects), np.shape(vibration_objects[0])

#sys.exit(0)


'''
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
count = 0
current = 0
step = 25600 
y_data = vibration_objects[0]
x_data = range(0, step)
print np.shape(y_data[count*step:(count+1)*step])


def animate(i):
	global count
	global current
	if ((count%20) == 0):
		current = count
	print count
	ax1.clear()
	#ax1.clear()
	x_data = range(current*step, (count+1)*step)
	ax1.plot(x_data, y_data[current*step:(count+1)*step])
	#ax1.plot(x_data,y_data[count*step:(count+1)*step])
	count +=1

ani = FuncAnimation(fig, animate, interval=500)#frames=np.linspace(0, 2*np.pi, 128), init_func=init, blit=True)
plt.show()
sys.exit(0)
'''
y_freq = 25600
channels = np.shape(vibration_objects)[0]
y = np.zeros((channels, y_freq))
Phi = [None] * channels
#y = [None] * y_freq

print np.shape(y), np.shape(y[0])
#sys.exit(0)

y_start = input("Start_index? => ")
y_end = y_start + y_freq * training_sets


for i in range(0, channels):
    y[i] = vibration_objects[i][y_start:y_end]

#rescale y to make original k-svd algo work properly again
t = training_sets

for j in range(0, channels):
    y_mp = np.reshape(y[i], (t,y_freq))
    g = np.shape(y_mp)[1]
    #Phi_designer_k_svd(Phi_test, y_mp,maxiter = max_iter)
    ksvd = ApproximateKSVD(n_components = dict_component, transform_n_nonzero_coefs=sparsity)
    y_cur = np.reshape(y_mp, (t * k,m))
    y_cur =  y_cur.T
    Phi[j] = ksvd.fit(y_cur).components_

print "Done reading in incoming signal/components"
#Phi = Phi_test

#note to self: try taking dictionary of the the normal vibrating spindle at point #350000, idle at 0


chan = 1
count = 0

for k in range(0, channels):
	a, b = np.shape(Phi[k])
	write_name = f_name + "/" + f_name + "_" + str(chan) + "_" + str(count) + ".h5"
	file_create(write_name, Phi[k], a, b)
	if ((k+1) % 4 == 0):
		count = 0
		chan += 1
	else:
		count +=1
	#count += 1
print "Done creating dictionaries for all channels"

sys.exit(0)
#file_create(f_name, Phi, a, b)