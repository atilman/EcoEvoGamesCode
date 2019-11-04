#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 11:56:20 2018

@author: atilman
"""

"""
Make plots for d11 d01 d10 d00 that show the different regimes as a function of the parameters
"""

import numpy as np
import pylab as plt
"""
Define color pallette
"""
from matplotlib.colors import LinearSegmentedColormap

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

blu=parula_map(55)
grn=parula_map(135)
yel=parula_map(245)
label01='$\Delta_H^1$'
label10='$\delta_L^0$'
"""
SUBFIGURE (c)
"""
d11=-1
d00=1

"Plots and Filling instructions"
fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1)
d01=np.arange(-4,4,.01)
y1=d01-2*(-d11*d00)**(1/2)
plt.plot(d01,y1,'k')
ax.fill_between(d01,y1,5,facecolor=blu)

d01=np.arange(np.sqrt(-d11*d00),4,.01)
y2=d11*d00/d01
y1=d01-2*(-d11*d00)**(1/2)
plt.plot(d01,y2,'k')

ax.fill_between(d01,y1, y2,facecolor=yel, interpolate=True)
ax.fill_between(d01,-5, y2, facecolor=grn, interpolate=True)

d01=np.arange(-4,np.sqrt(-d11*d00)+.01,.01)
y1=d01-2*(-d11*d00)**(1/2)
ax.fill_between(d01,y1, -5, facecolor=grn, interpolate=True)

#"contour"
#x = np.arange(np.sqrt(-d11*d00),4,.01)
#y = np.arange(-np.sqrt(-d11*d00),4,.01)
#d01, d10 = np.meshgrid(x,y)
#eCrit = (((4*d11*d00+(d01-d10)**2)**(1/2)-d10-d01)*((4*d11*d00+(d01-d10)**2)**(1/2)-2*d11+d01-d10)*((4*d11*d00+(d01-d10)**2)**(1/2)-2*d00-d01+d10))/(8*(d11+d10-d01-d00)**2)
#levels=[.1,.2,.3,.5]
#CS = ax.contour(d01, d10, eCrit,levels,colors = 'k')
#ax.clabel(CS,CS.levels[::2], fmt='%2.1f',fontsize=12, inline=1)

"General sytle for all subfigs"
plt.ylim(-4,4)
plt.xlim(-4,4)
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_yticks([])
ax.set_xticks([])
plt.xlabel(label01,fontsize=20)
plt.ylabel(label10,fontsize=20)
ax.xaxis.set_label_coords(1.07, 0.53)
ax.yaxis.set_label_coords(.52, 1.1)

plt.show()



"""
SUBFIGURE (b)
"""
d11=1
d00=1
"Plots and Filling instructions"

x = np.arange(.015,4,.01)
y = np.arange(.01,4,.01)



fig1 = plt.figure(2)
ax1 = fig1.add_subplot(1, 1, 1)

d01=np.arange(.01,4,.01)
y2=d11*d00/d01
plt.plot(d01,y2,'k')
ax1.fill_between(d01,5,y2,facecolor=yel)
ax1.fill_between(d01,-5,y2,facecolor=blu)
d01, d10 = np.meshgrid(x,y)
eCrit = (((4*d11*d00+(d01-d10)**2)**(1/2)-d10-d01)*((4*d11*d00+(d01-d10)**2)**(1/2)-2*d11+d01-d10)*((4*d11*d00+(d01-d10)**2)**(1/2)-2*d00-d01+d10))/(8*(d11+d10-d01-d00)**2)
levels=[.1,.2,.3,.4,.5,.6]
CS = ax1.contour(d01, d10, eCrit,levels,colors = 'k')
ax1.clabel(CS,CS.levels[::2], fmt='%2.1f',fontsize=12, inline=1)

d01=np.arange(-4,0,.01)
ax1.fill_between(d01,-5,5,facecolor=blu)

"General sytle for all subfigs"
plt.ylim(-4,4)
plt.xlim(-4,4)
ax1.spines['bottom'].set_position('zero')
ax1.spines['left'].set_position('zero')
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.set_yticks([])
ax1.set_xticks([])
plt.xlabel(label01,fontsize=20)
plt.ylabel(label10,fontsize=20)
ax1.xaxis.set_label_coords(1.07, 0.53)
ax1.yaxis.set_label_coords(.52, 1.1)

plt.show()


"""
SUBFIGURE (a)
"""
d11=-1
d00=-1

"Plots and Filling instructions"
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(1, 1, 1)
d01=np.arange(-4,4,.01)
ax2.fill_between(d01,-5,5,facecolor=grn)



"General sytle for all subfigs"
plt.ylim(-4,4)
plt.xlim(-4,4)
ax2.spines['bottom'].set_position('zero')
ax2.spines['left'].set_position('zero')
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.set_yticks([])
ax2.set_xticks([])
plt.xlabel(label01,fontsize=20)
plt.ylabel(label10,fontsize=20)
ax2.xaxis.set_label_coords(1.07, 0.53)
ax2.yaxis.set_label_coords(.52, 1.1)

plt.show()


"""
SUBFIGURE (d)
"""
d11=1
d00=-1
"Plots and Filling instructions"
fig3 = plt.figure(2)
ax3 = fig3.add_subplot(1, 1, 1)
d01=np.arange(-4,4,.01)
y1=d01+2*(-d11*d00)**(1/2)
plt.plot(d01,y1,'k')
ax3.fill_between(d01,y1,-5,facecolor=blu)

d01=np.arange(-4,-np.sqrt(-d11*d00)+.01,.01)
y1=d01+2*(-d11*d00)**(1/2)
ax3.fill_between(d01,5, y1, facecolor=grn, interpolate=True)

d01=np.arange(-np.sqrt(-d11*d00),0,.01)
y1=d01+2*(-d11*d00)**(1/2)
y2=d11*d00/d01
plt.plot(d01,y2,'k')
ax3.fill_between(d01,y2,5,facecolor=grn)
ax3.fill_between(d01,y2,y1,facecolor=yel)

d01=np.arange(0,4,.01)
y1=d01+2*(-d11*d00)**(1/2)
ax3.fill_between(d01,5,y1,facecolor=yel)


"General sytle for all subfigs"
plt.ylim(-4,4)
plt.xlim(-4,4)
ax3.spines['bottom'].set_position('zero')
ax3.spines['left'].set_position('zero')
ax3.spines['right'].set_color('none')
ax3.spines['top'].set_color('none')
ax3.set_yticks([])
ax3.set_xticks([])
plt.xlabel(label01,fontsize=20)
plt.ylabel(label10,fontsize=20)
ax3.xaxis.set_label_coords(1.07, 0.53)
ax3.yaxis.set_label_coords(.52, 1.1)

plt.show()




"""SAVE FIGS"""
#
#fig.savefig('../FIGS/d01d10clowres.png',dpi=100, bbox_inches='tight')
#fig1.savefig('../FIGS/d01d10blowres.png',dpi=100, bbox_inches='tight')
#fig2.savefig('../FIGS/d01d10alowres.png',dpi=100, bbox_inches='tight')
#fig3.savefig('../FIGS/d01d10dlowres.png',dpi=100, bbox_inches='tight')
#
#"""HRES FIGS"""
#fig.savefig('../FIGS/d01d10c.png',dpi=300, bbox_inches='tight')
#fig1.savefig('../FIGS/d01d10b.png',dpi=300, bbox_inches='tight')
#fig2.savefig('../FIGS/d01d10a.png',dpi=300, bbox_inches='tight')
#fig3.savefig('../FIGS/d01d10d.png',dpi=300, bbox_inches='tight')
#
#"""PDF FIGS"""
#fig.savefig('../FIGS/d01d10c1.pdf', bbox_inches='tight')
#fig1.savefig('../FIGS/d01d10b1.pdf', bbox_inches='tight')
#fig2.savefig('../FIGS/d01d10a1.pdf', bbox_inches='tight')
#fig3.savefig('../FIGS/d01d10d1.pdf', bbox_inches='tight')



























