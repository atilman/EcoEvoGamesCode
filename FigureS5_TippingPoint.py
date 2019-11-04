#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 16:39:44 2018

@author: atilman
"""


import numpy as np
from multiprocessing import Pool
from scipy.stats import norm
from scipy.stats import truncnorm
import pylab as plt
from scipy.integrate import solve_ivp


def dndx(W,t0,epsilon,mu,sig,d00,d10,d01,d11,l):
    """
    W[0]: Enviornmental state measure `n'
    W[1]: Fraction of high effort harvesters `x'
    """
    
    "Gradient of selection"
    g = (1-W[0])*(d10*W[1]+d00*(1-W[1]))-W[0]*(d11*W[1]+d01*(1-W[1]))
    f = truncnorm.ppf(W[0],a,b,mu,sig)
    h = norm.ppf(W[0],mu,sig)
    theta=1/mu - 1 
    "dynamical equations"
    dx = W[1]*(1-W[1])*(g)
    dn = [ epsilon*W[0]*(1-W[0])*(-1+(1+theta)*W[1]) , epsilon*W[0]*(1-W[0])*(W[1]-mu) , epsilon*W[0]*(1-W[0])*(W[1]-mu+sig/2-sig*W[0]), epsilon*W[0]*(1-W[0])*(W[1]-h) , epsilon*W[0]*(1-W[0])*(W[1]-f) , epsilon*(W[1]-f)  ]
  
    return([dn[l], dx])
  
def simRun(A):
    epsilon = A[0]
    mu = A[1]
    sig = A[2]
    d00 = A[3]
    d10 = A[4]
    d01 = A[5]
    d11 = A[6]
    W0 = A[7]
    tEnd = A[8]
    l = A[9]
    args = (epsilon,mu,sig,d00,d10,d01,d11,l)
    fun = lambda t,y: dndx(y,t,*args)
    sol = solve_ivp(fun,[0,tEnd],W0,rtol=10**-13,atol=10**-13)
    return(sol)
    
mu=1/2
epsilon = [1,1,2,2,.5,.5]
sig = [.1,.1,.3,.1,.3,.1]
   
myclip_a=0
myclip_b=1
my_mean=mu
draft=False 
d11 = 2;  "DtL"
d00 = 1; "dtH"
d10 = 3 ; "dtL"
d01 = 1; "DtH"
numSims = 4;
tEnd=200
viewRun=15


save = 0        
opacity=.08
initVarx=0.02
initVarn=0.05
initx=np.linspace(.1,.9,numSims)
initn=np.linspace(.1,.9,numSims)
W0=np.empty((numSims,numSims,2),dtype=object)
for i in range(numSims):
    for j in range(numSims):
        W0[i,j,:] = [initn[i]+np.random.normal(0,initVarn,1)[0], initx[j]+np.random.normal(0,initVarx,1)[0]]
modelName=['ExactWeitzModel','ourVariant','GenLinear','GenSigmoid','truncNorm','truncnormMod'] 

s = 0
for l in [0,1,2,4,2,4]:
    myclip_a=float(0)
    myclip_b=float(1)
    my_mean=float(mu)
    my_std=float(sig[s])
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    frac=np.linspace(0,1,1000)
    xNullcline=((d00)*(1-frac)+(d10)*frac)/((d01+d00)*(1-frac)+(d11+d10)*frac)
    xNullcline[xNullcline>1]=np.nan
    xNullcline[xNullcline<0]=np.nan
    nLin = frac/sig[s]-mu/sig[s]+1/2
    nLin[nLin>1]=np.nan
    nLin[nLin<0]=np.nan
    nNull = [norm.cdf(frac,mu,.00001) , norm.cdf(frac,mu,.00001), nLin , norm.cdf(frac,mu,sig[s]),truncnorm.cdf(frac,a,b,mu,sig[s]),truncnorm.cdf(frac,a,b,mu,sig[s])]
    args = (epsilon[s],mu,sig[s],d00,d10,d01,d11,l)
    fun=lambda t,y: dndx(y,t,*args)
    solNX=[]
    solT=[]
    A=np.empty((numSims*numSims,10),dtype=object)
    for i in range(numSims):
        for j in range(numSims):
            A[i*numSims+j]=(epsilon[s],mu,sig[s],d00,d10,d01,d11,W0[i,j,:],tEnd,l)
           
    s = s+1      
    if __name__=='__main__':
        with Pool(7) as pool:        
            output = pool.map(simRun,A)
            
 
    fig = plt.figure(figsize=(10,5))
    
    
    ax1 = fig.add_subplot(1,2,2)
    ax2 = fig.add_subplot(1,2,1)
    
    
    ax1.plot(output[viewRun].t,output[viewRun].y[0], 'b-',alpha=.71, label='Environmental state')
    ax1.plot(output[viewRun].t,output[viewRun].y[1], 'r-',alpha=.71, label='Low-impact fraction')
    ax1.set_ylim((0,1))
    
    ax1.set_title("Temporal Dynamics")
    ax1.set_xlabel("Time")
    ax1.grid()
    ax1.legend(loc='best')
        
    """
    SOLUTION CURVES
    """
    for i in range(numSims*numSims):
            if (d11<0 and d00<0) or (d11<0 and d00>0 and d01-d10>2*(-d11*d00)**.5) or (d11>0 and d00<0 and d01-d10<-2*(-d11*d00)**.5):
                if np.absolute(1-output[i].y[0][-1])<.01:
                    hue=1
                    ax2.plot(output[i].y[1], output[i].y[0], color=(.4, hue, 1-hue),alpha=.08,rasterized=draft)
                else:
                    hue=.4
                    ax2.plot(output[i].y[1], output[i].y[0], color=(0, hue, 1-hue),alpha=opacity,rasterized=draft)
                    
            else:
                ax2.plot(output[i].y[1], output[i].y[0], color='k',alpha=opacity,rasterized=draft)
               
    
    """
    NULLCLINES
    """
    if l != 5:
          ax2.plot([0,1],[0,0],color="blue")
          ax2.plot([0,1],[1,1],color="blue")
    ax2.plot(frac,nNull[l],color="blue",label='Environmental nullcline')
    ax2.plot(frac,xNullcline,color="r",label='Strategy nullclines')  
    ax2.plot([0,0],[0,1],color="r") 
    ax2.plot([1,1],[0,1],color="r") 
    ax2.set_ylim((-.05,1.05))
    ax2.set_xlim((-.05,1.05))
    ax2.set_xlabel("Fraction with low-impact strategy")
    ax2.set_ylabel("Environmental State")  
    ax2.set_title("Phase space")
    ax2.legend(loc=4)
    paramAnnote='epsilon='+str(epsilon)+', mu='+str(mu)+', sigma='+str(sig)+', d11='+str(d11)+', d00='+str(d00)+', d10='+str(d10)+', d01='+str(d01)+', tEnd='+str(tEnd)+', numSims='+str(numSims)+', Model='+modelName[l]
    if save==1:
        fig.savefig('../FIGS/'+'TIP'+modelName[l]+str(s)+'F.png',dpi=100, bbox_inches='tight')
        pars = plt.annotate(paramAnnote, (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top',fontsize=6)
        fig.savefig('../FIGS/'+'TIP'+modelName[l]+str(s)+'D.png',dpi=100,bbox_extra_artists=(pars,), bbox_inches='tight')












































