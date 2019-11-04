#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:06:33 2017

@author: atilman
"""

import numpy as np
import scipy.integrate as sp_int
import pylab as plt
"""

 both share info 
        """


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
 
frac=np.linspace(0,1,100)
K=4;
q=.5;
eL=.34
eH=.6
p=10;
w=5;
epsilon=2.5;
alphaL=.15
alphaH=.25
tEnd=6000
steps=20000
numSims=8
initVarR=0.05
initVarX=0.02
rSet=[.25,.29,.34,.37,.41,.46]
tEnd=[60,400,600,1000,150,30]
for i in range(len(rSet)):
    params = (rSet[i],K,q,eL,eH,p,w,epsilon,alphaL,alphaH)
    initR=np.linspace(.1,K/2,numSims)
    initX=np.linspace(.05,.95,numSims)
    W0 = [initR, initX]
    tRange=np.linspace(0,tEnd[i],steps)
    simData=np.zeros((len(initR),len(initX),len(tRange),2))
    for k in range(numSims):
        for j in range(numSims):
            W0 = [initR[k]+np.random.normal(0,initVarR,1)[0], initX[j]+np.random.normal(0,initVarX,1)[0]]
            simData[k,j,:,:] = sp_int.odeint(tragedy,y0=W0,t=tRange,args=params,rtol=1.49012e-12)
    fig = plt.figure(figsize=(5,5))
    ax2 = fig.add_subplot(1,1,1)
    for k in range(numSims):
        for j in range(numSims):
            ax2.plot(simData[k,j,:,1], simData[k,j,:,0], color="black",alpha=.07)
    res=K*(1-q/rSet[i]*(eL*frac*(1+alphaL*frac)+eH*(1-frac)*(1+alphaH*(1-frac))))
    strat=(w*(eH-eL))/(p*q*(eH+eH*alphaH-eL-frac*(eH*alphaH+eL*alphaL)))
    ax2.plot([0,1],[0.003,0.003],color="blue")
    ax2.plot(frac,res,color="blue",label='resource nullclines')
    ax2.plot(frac,strat,color="r",label='strategy nullclines')
    ax2.plot([0.003,0.003],[0,K],color="r")
    ax2.plot([.996,.996],[0,K],color="r")
    ax2.set_ylim((0,K/2))
    ax2.set_xlim((0,1))
    ax2.set_xlabel("Frequency of low-effort harvesters")
    ax2.set_ylabel("Resource level")  
    ax2.set_title("Phase space")
    ax2.legend(loc='upper right')
    fig.savefig('../FIGS/'+'infoShareEps'+str(epsilon).replace('.','')+'r'+str(rSet[i]).replace('.','')+'.png',dpi=150,bbox_inches='tight')
    
    











