#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:06:50 2018

@author: atilman
"""

import numpy as np
import pylab as plt
"Making a new figure 1."
ticks=(0,1)
tickL=("0","1")
fig = plt.figure(figsize=(8,8))
ax=plt.subplot(111)
#ax.plot([0,1],[0,1],"k--",linewidth=4,label='Environmental nullcline')
ax.set_ylim((0,1))
ax.set_xlim((0,1))
plt.yticks(ticks,tickL,fontsize=16)
plt.xticks(ticks,tickL,fontsize=16)
ax.set_ylabel('Environmental State, n',fontsize=24)
ax.set_xlabel('Low-impact strategy fraction, x',fontsize=24)
label00 = '$\delta_H^0$'
label01 = '$\Delta_H^1$'
label10 = '$\delta_L^0$'
label11 = '$\Delta_L^1$'




#plt.annotate('', xy=(.98, 1), xytext=(.8, 1),
#            arrowprops=dict(facecolor='b',ec='b',width=6,headwidth=20),
#            )
#plt.annotate('', xy=(.6, 1), xytext=(.8, 1),
#            arrowprops=dict(facecolor='r',ec='r',width=6,headwidth=20),
#            )
plt.text(1, 1, label11,
         {'color': 'k', 'fontsize': 30, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
#plt.text(.73,1,'+',
#         {'color': 'k', 'fontsize': 24, 'ha': 'center', 'va': 'center',
#          'bbox': dict(boxstyle="round", fc="w", ec="r", pad=0.1)})
#plt.text(.863,1,'-',
#         {'color': 'k', 'fontsize': 24, 'ha': 'center', 'va': 'center',
#          'bbox': dict(boxstyle="round", fc="w", ec="b", pad=0.15)})
#
#plt.annotate('', xy=(0.01, 1), xytext=(.2, 1),
#            arrowprops=dict(facecolor='r',ec='r',width=6,headwidth=20),
#            )
#plt.annotate('', xy=(.4, 1), xytext=(.2, 1),
#            arrowprops=dict(facecolor='b',ec='b',width=6,headwidth=20),
#            )
plt.text(0, 1, label01,
         {'color': 'k', 'fontsize': 30, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
#plt.text(.13,1,'+',
#         {'color': 'k', 'fontsize': 24, 'ha': 'center', 'va': 'center',
#          'bbox': dict(boxstyle="round", fc="w", ec="r", pad=0.1)})
#plt.text(.263,1,'-',
#         {'color': 'k', 'fontsize': 24, 'ha': 'center', 'va': 'center',
#          'bbox': dict(boxstyle="round", fc="w", ec="b", pad=0.15)})
#
#plt.annotate('', xy=(.99, 0), xytext=(.8, 0),
#            arrowprops=dict(facecolor='r',ec='r',width=6,headwidth=20),
#            )
#plt.annotate('', xy=(.6, 0), xytext=(.8, 0),
#            arrowprops=dict(facecolor='b',ec='b',width=6,headwidth=20),
#            )
plt.text(1, 0, label10,
         {'color': 'k', 'fontsize': 30, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
#plt.text(.74,0,'-',
#         {'color': 'k', 'fontsize': 24, 'ha': 'center', 'va': 'center',
#          'bbox': dict(boxstyle="round", fc="w", ec="b", pad=0.1)})
#plt.text(.573,0.03,'+',
#         {'color': 'k', 'fontsize': 24, 'ha': 'center', 'va': 'center',
#          'bbox': dict(boxstyle="round", fc="w", ec="r", pad=0.15)})
#
#plt.annotate('', xy=(0.11, .085), xytext=(.3, .085),
#            arrowprops=dict(facecolor='b',ec='b',width=6,headwidth=20),
#            )
#plt.annotate('', xy=(.3, .03), xytext=(.11, .03),
#            arrowprops=dict(facecolor='r',ec='r',width=6,headwidth=20),
#            )
plt.text(0, 0, label00,
         {'color': 'k', 'fontsize': 30, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
#plt.text(.14,0,'-',
#         {'color': 'k', 'fontsize': 24, 'ha': 'center', 'va': 'center',
#          'bbox': dict(boxstyle="round", fc="w", ec="b", pad=0.1)})
#plt.text(.2,.035,'+',
#         {'color': 'k', 'fontsize': 24, 'ha': 'center', 'va': 'center',
#          'bbox': dict(boxstyle="round", fc="w", ec="r", pad=0.15)})
#
#plt.annotate('', xy=(0, 0.4), xytext=(0, .6),
#            arrowprops=dict(facecolor='black',width=6,headwidth=20),
#            )
#plt.annotate('', xy=(1, 0.6), xytext=(1, .4),
#            arrowprops=dict(facecolor='black',width=6,headwidth=20),
#            )
#plt.annotate('', xy=(0, 0.8), xytext=(0, .98),
#            arrowprops=dict(facecolor='black',width=6,headwidth=20),
#            )
#plt.annotate('', xy=(1, 0.98), xytext=(1, .8),
#            arrowprops=dict(facecolor='black',width=6,headwidth=20),
#            )

#fig.savefig('../FIGS/figureOne3a.pdf',bbox_inches='tight')





