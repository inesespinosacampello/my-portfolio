# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:23:34 2023

@author: inese
"""

import numpy as np
import matplotlib.pyplot as plt
import calendar
from itertools import cycle

max_temp = np.loadtxt('C:\\Users\\inese\\OneDrive\\Escritorio\\max_temp.dat')
min_temp = np.loadtxt('C:\\Users\\inese\\OneDrive\Escritorio\\min_temp.dat')

avg_max = np.mean(max_temp[:,1:], axis=1)
avg_min = np.mean(min_temp[:,1:], axis=1)

bins = np.arange(0,20,1)

width = 0.8 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
hist, bins = np.histogram(avg_max, bins=bins)

plt.xlim((min(bins),max(bins)))
plt.ylim((0,1.25*max(hist)))
plt.xlabel('average max temperature (C)')
plt.ylabel('number of years')
plt.bar(center,hist,align='center',width=width)
plt.show()

hist, bins = np.histogram(avg_min, bins=bins)

plt.xlim((min(bins),max(bins)))
plt.ylim((0,1.25*max(hist)))
plt.xlabel('average min temperature (C)')
plt.ylabel('number of years')
plt.bar(center,hist,align='center',width=width)
plt.show()








markers = ['o','v','^','>','<','8','s','p','*','h','H','D']
skip = 10

max_temp = np.loadtxt('C:\\Users\\inese\\OneDrive\\Escritorio\\max_temp.dat')
min_temp = np.loadtxt('C:\\Users\\inese\\OneDrive\Escritorio\\min_temp.dat')

x = list(range(1,13))
xlab = [calendar.month_abbr[i] for i in range(1,13)]

plt.subplots_adjust(hspace = .001)
ax = plt.subplot(2, 1, 1)
plt.margins(0.1)
plt.xticks(x, xlab, rotation='vertical')
plt.yticks(fontsize=12)
plt.ylabel(r'$T_{min}({}^\circ C)$',fontsize=18)
plt.ylim((1.1*np.min(min_temp[:,1:]),1.1*np.max(max_temp[:,1:])))
frame = plt.gca()
frame.axes.xaxis.set_ticklabels([])
for (yr,temp,marker) in zip(min_temp[::skip,0],min_temp[::skip,1:],cycle(markers)):
    plt.plot(x,temp,label=str(int(yr)),marker=marker,ms=8,lw=1)
    

ax.legend(loc='upper right', bbox_to_anchor=(1.25, 0.6), title='year', fancybox=True, shadow=True)

plt.subplot(2, 1, 2)
plt.margins(0.1)
plt.xticks(x, xlab, rotation='vertical',fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('month',fontsize=18)
plt.ylabel(r'$T_{max}({}^\circ C)$',fontsize=18)
plt.ylim((1.1*np.min(min_temp[:,1:]),1.1*np.max(max_temp[:,1:])))

for (yr,temp,marker) in zip(max_temp[::skip,0],max_temp[::skip,1:],cycle(markers)):
    plt.plot(x,temp,label=str(int(yr)),marker=marker,ms=8,lw=1)
    
#plt.legend(loc='top left')

plt.show()