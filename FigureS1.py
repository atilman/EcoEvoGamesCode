#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:01:39 2019

@author: atilman
"""
import numpy as np
import cmath as cm
import scipy as sp
import scipy.integrate as sp_int
import pylab as plt
from sympy import Symbol, solve
from sympy.functions import re, im

def eqFinder(r,K,q,eL,eH,p,w,gamma):
    n=Symbol('n')
    A = solve(p*q*n-p*q*r*gamma*n**2+p*q*r*gamma*n**3/K-w,n)
    B = np.array(A)
    C = B.astype('complex128')
    nStar=np.zeros(len(C))
    for i in range(len(C)):
        if np.abs(im(C[i]))<.001:
            nStar[i] = re(C[i])
        else:
            nStar[i] = np.nan
    xStar = eH/(eH-eL) - (r)/(q*(eH-eL))*(1-1/K*nStar)
    return(xStar,nStar)
r=.3;
K=6;
q=.5;
p=25;
w=12;
bL=.5;
bH=.97;
eL=bL*r/q*(1-w/(p*q*K))
eH=bH*r/q*(1-w/(p*q*K))
eL=.252
eH=.49

gam=np.linspace(.01,3,50000)


nStar1 = K/3 - (K**2 - 3*K/(gam*r))/(3*(-K**3 + 9*K**2/(2*gam*r) - 27*K*w/(2*gam*p*q*r) + np.sqrt(-4*(K**2 - 3*K/(gam*r))**3 + (-2*K**3 + 9*K**2/(gam*r) - 27*K*w/(gam*p*q*r))**2 +0j)/2)**(1/3)) - (-K**3 + 9*K**2/(2*gam*r) - 27*K*w/(2*gam*p*q*r) + np.sqrt(-4*(K**2 - 3*K/(gam*r))**3 + (-2*K**3 + 9*K**2/(gam*r) - 27*K*w/(gam*p*q*r))**2 +0j)/2)**(1/3)/3
nStar2 = K/3 - (K**2 - 3*K/(gam*r))/(3*(-1/2 - np.sqrt(3)*1j/2)*(-K**3 + 9*K**2/(2*gam*r) - 27*K*w/(2*gam*p*q*r) + np.sqrt(-4*(K**2 - 3*K/(gam*r))**3 + (-2*K**3 + 9*K**2/(gam*r) - 27*K*w/(gam*p*q*r))**2 +0j)/2)**(1/3)) - (-1/2 - np.sqrt(3)*1j/2)*(-K**3 + 9*K**2/(2*gam*r) - 27*K*w/(2*gam*p*q*r) + np.sqrt(-4*(K**2 - 3*K/(gam*r))**3 + (-2*K**3 + 9*K**2/(gam*r) - 27*K*w/(gam*p*q*r))**2 +0j)/2)**(1/3)/3
nStar3 = K/3 - (K**2 - 3*K/(gam*r))/(3*(-1/2 + np.sqrt(3)*1j/2)*(-K**3 + 9*K**2/(2*gam*r) - 27*K*w/(2*gam*p*q*r) + np.sqrt(-4*(K**2 - 3*K/(gam*r))**3 + (-2*K**3 + 9*K**2/(gam*r) - 27*K*w/(gam*p*q*r))**2 +0j)/2)**(1/3)) - (-1/2 + np.sqrt(3)*1j/2)*(-K**3 + 9*K**2/(2*gam*r) - 27*K*w/(2*gam*p*q*r) + np.sqrt(-4*(K**2 - 3*K/(gam*r))**3 + (-2*K**3 + 9*K**2/(gam*r) - 27*K*w/(gam*p*q*r))**2 +0j)/2)**(1/3)/3

xStar1 = eH/(eH-eL) - (r)/(q*(eH-eL))*(1-nStar1/K)
xStar2 = eH/(eH-eL) - (r)/(q*(eH-eL))*(1-nStar2/K)
xStar3 = eH/(eH-eL) - (r)/(q*(eH-eL))*(1-nStar3/K)

xS1 = np.where(np.abs(xStar1.imag)<.001,xStar1.real,np.nan)
xS2 = np.where(np.abs(xStar2.imag)<.001,xStar2.real,np.nan)
xS3 = np.where(np.abs(xStar3.imag)<.001,xStar3.real,np.nan)

wid=4
fig1 = plt.figure(figsize=(8,5))
ax = fig1.add_subplot(1,1,1)
ax.plot(gam,xS1,'k-',lw=wid)
ax.plot(gam,xS2,'k--',lw=wid)    
ax.plot(gam,xS3,'k-',lw=wid)
ax.set_ylim((0,1))
ax.set_xlim((gam[0],gam[-1]))
x0Star = 0.001+0*gam
x1StarS = np.where(xS3>1,.99+0*gam,np.nan)
x1StarD = np.where(xS3>1,np.nan,.99+0*gam)
x0StarS = np.where(xS1<0,.001+0*gam,np.nan)
x0StarD = np.where(xS1<0,np.nan,.001+0*gam)
ax.plot(gam,x0StarD,'k--',lw=wid)
ax.plot(gam,x0StarS,'k-',lw=wid)
ax.plot(gam,x1StarD,'k--',lw=wid)
ax.plot(gam,x1StarS,'k-',lw=wid)
ax.set_xlabel('$\gamma$',fontsize=20)
ax.set_ylabel('$x^*$',fontsize=20)


r=.3;
K=6;
q=.5;
p=50;
w=12;
bL=.5;
bH=.97;
eL=bL*r/q*(1-w/(p*q*K))
eH=bH*r/q*(1-w/(p*q*K))
eL=.276
eH=.564

gam=np.linspace(.01,3,50000)


nStar1 = K/3 - (K**2 - 3*K/(gam*r))/(3*(-K**3 + 9*K**2/(2*gam*r) - 27*K*w/(2*gam*p*q*r) + np.sqrt(-4*(K**2 - 3*K/(gam*r))**3 + (-2*K**3 + 9*K**2/(gam*r) - 27*K*w/(gam*p*q*r))**2 +0j)/2)**(1/3)) - (-K**3 + 9*K**2/(2*gam*r) - 27*K*w/(2*gam*p*q*r) + np.sqrt(-4*(K**2 - 3*K/(gam*r))**3 + (-2*K**3 + 9*K**2/(gam*r) - 27*K*w/(gam*p*q*r))**2 +0j)/2)**(1/3)/3
nStar2 = K/3 - (K**2 - 3*K/(gam*r))/(3*(-1/2 - np.sqrt(3)*1j/2)*(-K**3 + 9*K**2/(2*gam*r) - 27*K*w/(2*gam*p*q*r) + np.sqrt(-4*(K**2 - 3*K/(gam*r))**3 + (-2*K**3 + 9*K**2/(gam*r) - 27*K*w/(gam*p*q*r))**2 +0j)/2)**(1/3)) - (-1/2 - np.sqrt(3)*1j/2)*(-K**3 + 9*K**2/(2*gam*r) - 27*K*w/(2*gam*p*q*r) + np.sqrt(-4*(K**2 - 3*K/(gam*r))**3 + (-2*K**3 + 9*K**2/(gam*r) - 27*K*w/(gam*p*q*r))**2 +0j)/2)**(1/3)/3
nStar3 = K/3 - (K**2 - 3*K/(gam*r))/(3*(-1/2 + np.sqrt(3)*1j/2)*(-K**3 + 9*K**2/(2*gam*r) - 27*K*w/(2*gam*p*q*r) + np.sqrt(-4*(K**2 - 3*K/(gam*r))**3 + (-2*K**3 + 9*K**2/(gam*r) - 27*K*w/(gam*p*q*r))**2 +0j)/2)**(1/3)) - (-1/2 + np.sqrt(3)*1j/2)*(-K**3 + 9*K**2/(2*gam*r) - 27*K*w/(2*gam*p*q*r) + np.sqrt(-4*(K**2 - 3*K/(gam*r))**3 + (-2*K**3 + 9*K**2/(gam*r) - 27*K*w/(gam*p*q*r))**2 +0j)/2)**(1/3)/3

xStar1 = eH/(eH-eL) - (r)/(q*(eH-eL))*(1-nStar1/K)
xStar2 = eH/(eH-eL) - (r)/(q*(eH-eL))*(1-nStar2/K)
xStar3 = eH/(eH-eL) - (r)/(q*(eH-eL))*(1-nStar3/K)

xS1 = np.where(np.abs(xStar1.imag)<.001,xStar1.real,np.nan)
xS2 = np.where(np.abs(xStar2.imag)<.001,xStar2.real,np.nan)
xS3 = np.where(np.abs(xStar3.imag)<.001,xStar3.real,np.nan)

wid=4
fig2 = plt.figure(figsize=(8,5))
ax1 = fig2.add_subplot(1,1,1)
ax1.plot(gam,xS1,'k-',lw=wid)
ax1.plot(gam,xS2,'k--',lw=wid)    
ax1.plot(gam,xS3,'k-',lw=wid)
ax1.set_ylim((0,1))
ax1.set_xlim((gam[0],gam[-1]))
x0Star = 0.001+0*gam
x1StarS = np.where(xS3>1,.99+0*gam,np.nan)
x1StarD = np.where(xS3>1,np.nan,.99+0*gam)
x0StarS = np.where(xS1<0,.001+0*gam,np.nan)
x0StarD = np.where(xS1<0,np.nan,.001+0*gam)
ax1.plot(gam,x0StarD,'k--',lw=wid)
ax1.plot(gam,x0StarS,'k-',lw=wid)
ax1.plot(gam,x1StarD,'k--',lw=wid)
ax1.plot(gam,x1StarS,'k-',lw=wid)
ax1.set_xlabel('$\gamma$',fontsize=20)
ax1.set_ylabel('$x^*$',fontsize=20)
   

#fig2.savefig('../FIGS/FigureS1a.pdf', bbox_inches='tight')
#fig1.savefig('../FIGS/FigureS1b.pdf', bbox_inches='tight')
#fig2.savefig('../FIGS/FigureS1a.png', bbox_inches='tight',dpi=180)
#fig1.savefig('../FIGS/FigureS1b.png', bbox_inches='tight',dpi=180)


































 
