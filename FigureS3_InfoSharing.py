#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:06:33 2017

@author: atilman
"""

import numpy as np
import scipy as sp
import scipy.integrate as sp_int
import pylab as plt
"""

 both share info 
        """
vf=0


def tragedy(W,t0,r,K,q0,eL,eH,p,w,epsilon,alphaL,alphaH):
    """
    W[0]: Resource biomass
    W[1]: Fraction of high effort harvesters
    """
    
    qL=q0*(1+alphaL*W[1])
    qH=q0*(1+alphaH*(1-W[1]))
    piH=(p*qH*W[0]-w)*eH
    piL=(p*qL*W[0]-w)*eL
    H=W[0]*(W[1]*qL*eL+(1-W[1])*qH*eH)
    dn = epsilon*(W[0]*r*(1 - W[0]/K)-H)
    dx = W[1]*(1-W[1])*(piL-piH)
    
    return([dn,dx])

r=.33;
K=4;
q=.5;
eL=.33
eH=.6
p=10;
w=5;
epsilon=1.3;

alphaL=.22;
alphaH=0.05;



tEnd=300
steps=20000
numSims=12
initVarR=0.05
initVarX=0.02

params = (r,K,q,eL,eH,p,w,epsilon,alphaL,alphaH)
initR=np.linspace(.1,K/2,numSims)
initX=np.linspace(.05,.95,numSims)
W0 = [initR, initX]
tRange=np.linspace(0,tEnd,steps)
simData=np.zeros((len(initR),len(initX),len(tRange),2))
for i in range(numSims):
    for j in range(numSims):
        W0 = [initR[i]+np.random.normal(0,initVarR,1)[0], initX[j]+np.random.normal(0,initVarX,1)[0]]
        simData[i,j,:,:] = sp_int.odeint(tragedy,y0=W0,t=tRange,args=params,rtol=1.49012e-12)

fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(1,2,2)
ax2 = fig.add_subplot(1,2,1)
Start=0
End=20000
ax1.plot(tRange[Start:End],simData[1,1,:,0][Start:End], 'b-', label='resource')
ax1.plot(tRange[Start:End],simData[1,1,:,1][Start:End], 'r-', label='low effort fraction')
ax1.set_ylim((0,K/2))
ax1.set_title("Temporal dynamics")
ax1.set_xlabel("time")
ax1.grid()
ax1.legend(loc='upper right')


frac=np.linspace(0,1,100)
res=K*(1-q/r*(eL*frac*(1+alphaL*frac)+eH*(1-frac)*(1+alphaH*(1-frac))))
strat=(w*(eH-eL))/(p*q*(eH+eH*alphaH-eL-frac*(eH*alphaH+eL*alphaL)))




for i in range(numSims):
    for j in range(numSims):
        ax2.plot(simData[i,j,:,1], simData[i,j,:,0], color="black",alpha=.07)
ax2.plot([0,1],[0,0],color="blue")
ax2.plot(frac,res,color="blue",label='resource nullclines')
ax2.plot(frac,strat,color="r",label='strategy nullclines')
ax2.plot([0.004,0.004],[0,K],color="r")
ax2.plot([.996,.996],[0,K],color="r")
ax2.set_ylim((0,K/2))
ax2.set_xlim((0,1))
ax2.set_xlabel("Fraction of low effort harvesters")
ax2.set_ylabel("Resource")  
ax2.set_title("Phase space")
ax2.legend(loc='upper right')
if vf==1:
    W = np.meshgrid(initR,initX)
    dW=tragedy(W,0,r,K,q,eL,eH,p,w,epsilon,alphaL,alphaH)
    ax2.quiver(W[1], W[0], dW[1], dW[0],angles='xy', scale_units='xy', scale = 3)



#fig.savefig('../FIGS/'+'infoShareEps'+str(epsilon).replace('.','')+'.png',dpi=90,bbox_inches='tight')
#fig.savefig('../FIGS/'+'cycle'+'.png', dpi=figDPI, bbox_extra_artists=(pars,), bbox_inches='tight')














