#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 16:39:44 2018

@author: atilman
"""

from multiprocessing import Pool
import numpy as np
import pylab as plt
from scipy.integrate import solve_ivp

def dndx(W,t0,epsilon,d00,d10,d01,d11):
    """
    W[0]: Enviornmental state measure `n'
    W[1]: Fraction of high effort harvesters `x'
    """
    
    "Gradient of selection"
    g = (1-W[0])*(d10*W[1]+d00*(1-W[1]))-W[0]*(d11*W[1]+d01*(1-W[1]))
        
    "dynamical equations"
    dx = W[1]*(1-W[1])*(g)
    dn = epsilon*(W[1]-W[0])
  
    return([dn, dx])
    
def simRun(A):
    epsilon = A[0]
    d00 = A[1]
    d10 = A[2]
    d01 = A[3]
    d11 = A[4]
    W0 = A[5]
    tEnd = A[6]
    args = (epsilon,d00,d10,d01,d11)
    fun = lambda t,y: dndx(y,t,*args)
    sol = solve_ivp(fun,[0,tEnd],W0,rtol=10**-13,atol=10**-13)
    return(sol)
    
    
RUN=2;  
epsilon=[30/100,261/1000,250/1000,235/1000];
draft=False 
d11 = -1/8;  "DtL"
d00 = 2; "dtH"
d10 = 1 ; "dtL"
d01 = 4; "DtH"
numSims = 10;
tEnd=[200,200,200,1000]
viewRun=50


paramAnnote='epsilon='+str(epsilon)+', d11='+str(d11)+', d00='+str(d00)+', d10='+str(d10)+', d01='+str(d01)+', tEnd='+str(tEnd)+', numSims='+str(numSims)           
opacity=.08
opacityGreen=[.28,.28,.28,.08]
initVarx=0.02
initVarn=0.05
initx=np.linspace(.1,.9,numSims)
initn=np.linspace(.1,.9,numSims)
A=np.empty((numSims*numSims,7),dtype=object)
for o in range(len(epsilon)):
    for i in range(numSims):
        for j in range(numSims):
            W0 = [initn[i]+np.random.normal(0,initVarn,1)[0], initx[j]+np.random.normal(0,initVarx,1)[0]]
            A[i*numSims+j]=(epsilon[o],d00,d10,d01,d11,W0,tEnd[o])
    
    if __name__=='__main__':
        with Pool(7) as pool:        
            output = pool.map(simRun,A)
            
            
    fig1 = plt.figure(figsize=(5,5))
    ax0 = fig1.add_subplot(1,1,1)
    
    
    
    frac=np.linspace(0,1,1000)
    xNullcline=((d00)*(1-frac)+(d10)*frac)/((d01+d00)*(1-frac)+(d11+d10)*frac)
    xNullcline[xNullcline>2]=np.nan
    xNullcline[xNullcline<-2]=np.nan

    
    
    
    for i in range(numSims*numSims):
            if (d11<0 and d00<0) or (d11<0 and d00>0 and d01-d10>2*(-d11*d00)**.5) or (d11>0 and d00<0 and d01-d10<-2*(-d11*d00)**.5):
                if np.absolute(1-output[i].y[0][-1])<.01:
                    hue=1
                    ax0.plot(output[i].y[1], output[i].y[0], color=(.4, hue, 1-hue),alpha=opacityGreen[o],rasterized=draft)
                else:
                    hue=.4
                    ax0.plot(output[i].y[1], output[i].y[0], color=(0, hue, 1-hue),alpha=opacity,rasterized=draft)
            else:
                ax0.plot(output[i].y[1], output[i].y[0], color='k',alpha=opacity,rasterized=draft)

    
    
    ax0.plot([0,1],[0,1],color="blue",label='Environmental nullcline') 
    ax0.plot(frac,xNullcline,color="r",label='Strategy nullclines') 
    ax0.plot([0.004,0.004],[0,1],color="r") 
    ax0.plot([.996,.996],[0,1],color="r") 
    ax0.set_ylim((0,1))
    ax0.set_xlim((0,1))
    ax0.set_xlabel("Fraction with low-impact strategy")
    ax0.set_ylabel("Environmental State")  
    ax0.set_title("Phase space")
    ax0.legend(loc='best')
    #fig1.savefig('../FIGS/'+'DecayFig4'+'Eps'+str(epsilon[o]).replace('.','')+'.png',dpi=180)
    #fig1.savefig('../FIGS/'+'DecayFig4'+'Eps'+str(epsilon[o]).replace('.','')+'.pdf')
eCrit = (((4*d11*d00+(d01-d10)**2)**(1/2)-d10-d01)*((4*d11*d00+(d01-d10)**2)**(1/2)-2*d11+d01-d10)*((4*d11*d00+(d01-d10)**2)**(1/2)-2*d00-d01+d10))/(8*(d11+d10-d01-d00)**2)
print(eCrit)










































