#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:28:32 2017

@author: atilman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:13:36 2017

@author: atilman
"""

import numpy as np
import scipy.integrate as sp_int
import pylab as plt
from sympy import Symbol, solve

"""
Choose a case:
    0: p fixed
    1: P a linearly decreasing fn of Harvest
        """
case=1


def tragedy(W,t0,r,K,q,eL,eH,p,w,epsilon,gamma):
    """
    W[0]: Resource biomass
    W[1]: Fraction of low effort harvesters
    """
    if case==0:
        piH=(p*q*W[0]-w)*eH
        piL=(p*q*W[0]-w)*eL
    else:
        mktPrice=p*(1-gamma*W[0]*q*(W[1]*eL+(1-W[1])*eH))
        piH=(mktPrice*q*W[0]-w)*eH
        piL=(mktPrice*q*W[0]-w)*eL
    dR = epsilon*(W[0]*r*(1 - W[0] / K)-q*W[0]*(W[1]*eL+(1-W[1])*eH))
    df = W[1]*(1-W[1])*(piL-piH)
  
    return([dR, df])

def eqFinder(r,K,q,eL,eH,p,w,gamma):
    x=Symbol('x')
    frac=solve((p-(p*(p-4*gamma*w*(x*eL+(1-x)*eH)))**(1/2))/(2*p*gamma*q*(x*eL+(1-x)*eH))-K*(1 - q/r*(x*eL + (1 - x)*eH)))
    frac1=solve((p+(p*(p-4*gamma*w*(x*eL+(1-x)*eH)))**(1/2))/(2*p*gamma*q*(x*eL+(1-x)*eH))-K*(1 - q/r*(x*eL + (1 - x)*eH)))
    n=np.zeros(len(frac))
    n1=np.zeros(len(frac1))
    for i in range(len(frac)):
        frac[i]=complex(frac[i])
        n[i]=K*(1-q/r*(frac[i]*eL+(1-frac[i])*eH))
    for i in range(len(frac1)):
        frac1[i]=complex(frac1[i])
        n1[i]=K*(1-q/r*(frac1[i]*eL+(1-frac1[i])*eH))
    return(frac,n,frac1,n1)

def EQ(M):
    r=M[0]
    K=M[1]
    q=M[2]
    eL=M[3]
    eH=M[4]
    p=M[5]
    w=M[6]
    gamma=M[8]
    x=Symbol('x')
    frac=solve((p-(p*(p-4*gamma*w*(x*eL+(1-x)*eH)))**(1/2))/(2*p*gamma*q*(x*eL+(1-x)*eH))-K*(1 - q/r*(x*eL + (1 - x)*eH)))
    frac1=solve((p+(p*(p-4*gamma*w*(x*eL+(1-x)*eH)))**(1/2))/(2*p*gamma*q*(x*eL+(1-x)*eH))-K*(1 - q/r*(x*eL + (1 - x)*eH)))
    n=np.zeros(len(frac))
    n1=np.zeros(len(frac1))
    for i in range(len(frac)):
        frac[i]=complex(frac[i])
        n[i]=K*(1-q/r*(frac[i]*eL+(1-frac[i])*eH))
    for i in range(len(frac1)):
        frac1[i]=complex(frac1[i])
        n1[i]=K*(1-q/r*(frac1[i]*eL+(1-frac1[i])*eH))

    return(frac,n,frac1,n1)

#%%
r=.3;
K=4;
q=.5;
p=35;
w=20;
epsilon=20;
gamma=1
eL=.25;
eH=.5;

tEnd=200
steps=40000
numSims=15
initVarR=0.045
initVarX=0.015
params=np.zeros((8,9))
params[0] = (r,K,q,eL,eH,p,10,epsilon,.1)
params[1] = (r,K,q,eL,eH,p,12,epsilon,1.3)
params[2] = (r,K,q,eL,eH,40,10,epsilon,2.5)
params[3] = (r,K,q,eL,eH,p,10,epsilon,2.9)
params[4] = (.36,39.6,.59,.357,.6,13.7,21.7,epsilon,.28)
params[5] = (r,K,q,eL,eH,p,5,epsilon,3.2)
params[6] = (.36,39.6,.59,.185,.515,13.7,16.6,epsilon,.28)
params[7] = (r,K,q,eL,eH,p,43,epsilon,0.001)

initX=np.linspace(.04,.96,numSims)
#W0 = [initR, initX]
tRange=np.linspace(0,tEnd,steps)
simData=np.zeros((8,numSims,numSims,len(tRange),2))
for k in range(8):
    for i in range(numSims):
        for j in range(numSims):
            initR=np.linspace(.3,params[k,1]/1.4,numSims)
            W0 = [initR[i]+np.random.normal(0,initVarR,1)[0], initX[j]+np.random.normal(0,initVarX,1)[0]]
            simData[k,i,j,:,:] = sp_int.odeint(tragedy,y0=W0,t=tRange,args=tuple(params[k]))



#%%

fig = plt.figure(figsize=(14,7))
frac=np.linspace(0,1,100000)
stableLeft=['k','w','w','w','w','k','k','w']
stableRight=['w','w','w','k','k','k','w','k']
stableLowArm=[['b'],'k',['b'],['b'],'k',['b'],['b'],['b']]
stableHighArm=[['b'],['b'],'k',['b','b','b'],['w','b'],['w','b','b','b'],['w','k','b','b'],['b']]
for k in range(8):
    ax = fig.add_subplot(2,4,k+1)
    for i in range(len(initR)):
        for j in range(len(initX)):
            ax.plot(simData[k,i,j,:,1], simData[k,i,j,:,0], color="black",alpha=.1)
    r=params[k,0]
    K=params[k,1]
    q=params[k,2]
    p=params[k,5]
    w=params[k,6]
    gam=params[k,8]
    eL=params[k,3]
    eH=params[k,4]
    epsilon=params[k,7]      
    Res=(p-np.sqrt(p*(p-4*gam*w*(frac*eL+(1-frac)*eH))))/(2*p*gam*q*(frac*eL+(1-frac)*eH))
    Res1=(p+np.sqrt(p*(p-4*gam*w*(frac*eL+(1-frac)*eH))))/(2*p*gam*q*(frac*eL+(1-frac)*eH))
    
    ax.plot([0,1],[0,0],color="blue")
    ax.plot([0,1],[K*(1-q*eH/r),K*(1-q*eL/r)],color="blue")
    ax.plot(frac,Res,color="r")
    ax.plot(frac,Res1,color="r")
    ax.plot([0.004,0.004],[0,K],color="r")
    ax.plot([.996,.996],[0,K],color="r")
    ax.set_ylim((0,K/1.4))
    ax.set_xlim((0,1))
    ax.set_yticks([])
    ax.set_xticks([]) 
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("n")
    A=eqFinder(r,K,q,eL,eH,p,w,gam)

    for i in range(len(A[0])):
        if abs(A[0][i].imag)<.0001:
            ax.scatter([A[0][i].real],[A[1][i]],s=120, facecolors=stableLowArm[k][i], edgecolors='k',zorder=20)
    for i in range(len(A[2])):
        if abs(A[2][i].imag)<.0001:
             ax.scatter([A[2][i].real],[A[3][i]],s=120, facecolors=stableHighArm[k][i], edgecolors='k',zorder=20)
    ax.scatter([0.01,0,.99,1],[0.01,K*(1-q*eH/r),0.01,K*(1-q*eL/r)],s=120, facecolors=['w',stableLeft[k],'w',stableRight[k]], edgecolors='k',zorder=20)    
    

#fig.savefig('../FIGS/'+'mktPricePanelS2'+'.png',dpi=150,bbox_inches='tight')















