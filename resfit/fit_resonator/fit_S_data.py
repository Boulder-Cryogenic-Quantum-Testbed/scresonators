# -*- coding: utf-8 -*-
"""
Created on Fri May 11 19:37:41 2018

@author: hung93
"""
import numpy as np
import lmfit
import matplotlib.pyplot as plt
import dataclasses
import pandas as pd
from matplotlib.gridspec import GridSpec
import sympy as sym
from scipy.optimize import curve_fit
from matplotlib.patches import Circle
from lmfit import Minimizer
import inflect
import matplotlib.pylab as pylab
from scipy import optimize
from scipy import stats
import time
import sys
import os
import attr
from typing import Tuple

from scipy.interpolate import interp1d

from resfit.fit_resonator.resonator import Resonator
import resfit.fit_resonator.fit_functions as ff

params = {'legend.fontsize': 10,
          'figure.figsize': (10, 8),
         'axes.labelsize': 18,
         'axes.titlesize':18,
         'xtick.labelsize':18,
         'ytick.labelsize':18,
         'lines.markersize' : 1,
         'lines.linewidth' : 2,
         'font.size': 15.0 }
pylab.rcParams.update(params)

np.set_printoptions(precision=4,suppress=True)
p = inflect.engine() # search ordinal
#####################################################################
## Data related
#####################################################################
def convert_params(from_method,params):
    if from_method =='DCM':
        Qc = params[2]/np.cos(params[4])
        Qi = params[1]*Qc/(Qc-params[1])
        Qc_INV = params[2]
        Qi_INV = Qi/(1+np.sin(params[4])/Qc_INV/2)
        return [1/params[0],Qi_INV,Qc_INV,params[3],-params[4],-params[5]]
    elif from_method == 'INV':
        Qc_DCM = params[2]
        Q_DCM = ( np.cos(params[4])/params[2]+1/params[1])**-1
        return [1/params[0],Q_DCM,Qc_DCM,params[3],-params[4],-params[5]]

#########################################################################

def Find_Circle(x,y):
    #Given a set of x,y data return a circle that fits data using LeastSquares Circle Fit Randy Bullock (2017)
    N = 0
    xavg = 0
    yavg = 0
    for i in range(0,len(x)):
        N = N + 1
        xavg = xavg + x[i]
    for i in range(0,len(y)):
        yavg = yavg + y[i]

    xavg = xavg/N
    yavg = yavg/N

    xnew = []
    ynew = []
    Suu = 0
    Svv = 0
    for i in range(0,len(x)):
        xnew.append(x[i]-xavg)
        Suu = Suu + (x[i]-xavg)*(x[i]-xavg)
    for i in range(0,len(y)):
        ynew.append(y[i]-yavg)
        Svv = Svv + (y[i]-yavg)*(y[i]-yavg)

    Suv = 0
    Suuu = 0
    Svvv = 0
    Suvv = 0
    Svuu = 0
    for i in range(0,len(xnew)):
        Suv = Suv + xnew[i] * ynew[i]
        Suuu = Suuu + xnew[i] * xnew[i] * xnew[i]
        Svvv = Svvv + ynew[i] * ynew[i] * ynew[i]
        Suvv = Suvv + xnew[i] * ynew[i] * ynew[i]
        Svuu = Svuu + ynew[i] * xnew[i] * xnew[i]
    Suv2 = Suv

    matrix1 = 0.5 * (Suuu + Suvv)
    matrix2 = 0.5 * (Svvv + Svuu)

    #row reduction for row 1
    Suv = Suv/Suu
    matrix1 = matrix1/Suu

    #row subtraction for row 2 by row 1
    Svv = Svv - (Suv * Suv2)
    matrix2 = matrix2 - (Suv2 * matrix1)

    #row reduction for row 2
    matrix2 = matrix2/Svv

    #row subtraction for row 1 by row 2
    matrix1 = matrix1 - (Suv * matrix2)

    #at this point matrix1 is x_c and matrix2 is y_c
    alpha = (matrix1*matrix1) + (matrix2*matrix2) + (Suu + Svv)/N
    R = alpha**(0.5)

    matrix1 = matrix1 + xavg
    matrix2 = matrix2 + yavg

    return matrix1, matrix2, R

#########################################################################

def Find_initial_guess(x,y1,y2,Method):#,output_path):
    try:
        y = y1 +1j*y2 #recombine transmission S21 from real and complex parts
        if Method.method == 'INV': #inverse transmission such that y = S21^(-1)
            y = 1/y
        y1 = np.real(y) #redefine y1 and y2 to account for possibility they were inversed above
        y2 = np.imag(y)
    except:
        raise ValueError(">Problem initializing data in Find_initial_guess(), please make sure data is of correct format")

    try:
        x_c,y_c,r = Find_Circle(y1,y2) #find circle that matches the data
        z_c = x_c+1j*y_c #define complex number to house circle center location data
    except:
        raise ValueError(">Problem in function Find_Circle, please make sure data is of correct format")

    # try:
    #     # plot(np.real(y),np.imag(y),"circle",output_path,np.real(z_c),np.imag(z_c),r)
    # except:
    #     print(">Error when trying to plot raw data and circle fit in Find_initial_guess")
    #     quit()

    try:
        ## move gap of circle to (0,0)
        ydata = y-1 #Theoretically should center point P at (0,0)
        z_c = z_c -1 #Shift guide circle to match data shift
    except:
        raise ValueError(">Error when trying to shift data into canonical position minus 1")

    try:
        #determine the angle to the center of the fitting circle from the origin
        if Method.method == 'INV':
            phi = np.angle(z_c)  ###########4
        else:
            phi = np.angle(-z_c)  ###########4

        freq_idx = np.argmax(np.abs(ydata))
        f_c = x[freq_idx] ###############3

        # plot(np.real(ydata),np.imag(ydata),"resonance",output_path,np.real(z_c),np.imag(z_c),r,np.real(ydata[freq_idx]),np.imag(ydata[freq_idx]))#plot data with guide circle
        # rotate resonant freq to minimum
        ydata = ydata*np.exp(-1j*phi) #should rotate the circle such that resonance is on the real axis

        z_c = z_c*np.exp(-1j*phi)
        # plot(np.real(ydata),np.imag(ydata),"phi",output_path,np.real(z_c),np.imag(z_c),r,np.real(ydata[freq_idx]),np.imag(ydata[freq_idx]))#plot shifted data with guide circle
    except:
        raise ValueError(">Error when trying to shift data according to phi in Find_initial_guess")

    try:
        if f_c < 0:
            raise ValueError(">Resonance frequency is negative. Please only input positive frequencies.")
    except:
        raise ValueError(">Cannot find resonance frequency in Find_initial_guess")

    if Method.method == 'DCM' or Method.method == 'PHI':
        try:
            Q_Qc = np.max(np.abs(ydata)) #diameter of the circle found from getting distance from (0,0) to resonance frequency data point (possibly should be using fit circle)
            y_temp =np.abs(np.abs(ydata)-np.max(np.abs(ydata))/2**0.5) #y_temp = |ydata|-(diameter/sqrt(2))

            _,idx1 = find_nearest(y_temp[0:freq_idx],0) #find min value in y_temp on one half of circle from resonance frequency
            _,idx2 = find_nearest(y_temp[freq_idx:],0) #find min value in y_temp on other half of circle from resonance frequency
            idx2 = idx2+freq_idx #add index of resonance frequency to get correct index for idx2
            kappa = abs((x[idx1] - x[idx2])) #bandwidth of frequencies (need 2pi?)

            Q = f_c/kappa
            Qc = Q/Q_Qc
            popt, pcov = curve_fit(One_Cavity_peak_abs, x,np.abs(ydata),p0 = [Q,Qc,f_c],bounds = (0,[np.inf]*3)) #fits parameters for the 3 terms given in p0 (this is where Qi and Qc are actually guessed)
            Q = popt[0]
            Qc = popt[1]
            init_guess = [Q,Qc,f_c,phi]
        except:
            if Method.method == 'DCM':
                raise ValueError(">Failed to find initial guess for method DCM. Please manually initialize a guess")
            else:
                raise ValueError(">Failed to find initial guess for method PHI. Please manually initialize a guess")

    elif Method.method == 'DCM REFLECTION':
        try:
            Q_Qc = np.max(np.abs(ydata))/2 #diameter of the circle found from getting distance from (0,0) to resonance frequency data point (possibly should be using fit circle)
            y_temp =np.abs(np.abs(ydata)-np.max(np.abs(ydata))/2**0.5) #y_temp = |ydata|-(diameter/sqrt(2))

            _,idx1 = find_nearest(y_temp[0:freq_idx],0) #find min value in y_temp on one half of circle from resonance frequency
            _,idx2 = find_nearest(y_temp[freq_idx:],0) #find min value in y_temp on other half of circle from resonance frequency
            idx2 = idx2+freq_idx #add index of resonance frequency to get correct index for idx2
            kappa = abs((x[idx1] - x[idx2])) #bandwidth of frequencies (need 2pi?)

            Q = f_c/kappa
            Qc = Q/Q_Qc
            popt, pcov = curve_fit(One_Cavity_peak_abs_REFLECTION, x,np.abs(ydata),p0 = [Q,Qc,f_c],bounds = (0,[np.inf]*3)) #fits parameters for the 3 terms given in p0 (this is where Qi and Qc are actually guessed)
            Q = popt[0]
            Qc = popt[1]
            init_guess = [Q,Qc,f_c,phi]
        except:
            raise ValueError(">Failed to find initial guess for method DCM REFLECTION. Please manually initialize a guess")

    elif Method.method == 'INV':

        try:
            Qi_Qc = np.max(np.abs(ydata)) #diameter of the circle found from getting distance from (0,0) to resonance frequency
            y_temp =np.abs(np.abs(ydata)-np.max(np.abs(ydata))/2**0.5) #y_temp = |ydata|-(diameter/sqrt(2))

            _,idx1 = find_nearest(y_temp[0:freq_idx],0) #find min value in y_temp on one half of circle from resonance frequency
            _,idx2 = find_nearest(y_temp[freq_idx:],0) #find min value in y_temp on other half of circle from resonance frequency
            idx2 = idx2+freq_idx #add index of resonance frequency to get correct index for idx2
            kappa = abs((x[idx1] - x[idx2])) #bandwidth of frequencies (need 2pi?)

            Qi = f_c/(kappa)
            Qc = Qi/Qi_Qc
            popt, pcov = curve_fit(One_Cavity_peak_abs, x,np.abs(ydata),p0 = [Qi,Qc,f_c],bounds = (0,[np.inf]*3)) #fits parameters for the 3 terms given in p0 (this is where Qi and Qc are actually guessed)
            Qi = popt[0]
            Qc = popt[1]
            init_guess = [Qi,Qc,f_c,phi]
        except:
            raise ValueError(">Failed to find initial guess for method INV. Please manually initialize a guess")

    elif Method.method == 'CPZM':
        #print(">Method CPZM not yet functional, please try another method")
        #quit()
        try:
            Q_Qc = np.max(np.abs(ydata))
            y_temp =np.abs(np.abs(ydata)-np.max(np.abs(ydata))/2**0.5)

            _,idx1 = find_nearest(y_temp[0:freq_idx],0)
            _,idx2 = find_nearest(y_temp[freq_idx:],0)
            idx2 = idx2+freq_idx
            kappa = abs((x[idx1] - x[idx2]))

            Q = f_c/kappa
            Qc = Q/Q_Qc
            popt, pcov = curve_fit(One_Cavity_peak_abs, x,np.abs(ydata),p0 = [Q,Qc,f_c],bounds = (0,[np.inf]*3))
            Q = popt[0]
            Qc = popt[1]
            Qa = -1/np.imag(Qc**-1*np.exp(-1j*phi)) #correct order?
            Qc = 1/np.real(Qc**-1*np.exp(-1j*phi))
            Qi = (1/Q-1/Qc)**-1
            Qic = Qi/Qc
            Qia = Qi/Qa
            init_guess = [Qi,Qic,f_c,Qia,kappa]
        except:
            raise ValueError(">Failed to find initial guess for method CPZM. Please manually initialize a guess")
    else:
        raise ValueError(">Method is not defined. Please choose a method: DCM, DCM REFLECTION, PHI, INV or CPZM")
    return init_guess,x_c,y_c,r

########################################################################
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    val = array[idx]
    return val, idx

###########################################################
   ## PLOT REAL and IMAG figure
def PlotFit(x,y,x_initial,y_initial,slope,intercept,slope2,intercept2,params,Method,func,error,figurename,x_c,y_c,radius,output_path,conf_array,extract_factor = None,title = "Fit",manual_params = None):
    plt.close(figurename) #close plot if still open
    #generate an even distribution of 5000 frequency points between the min and max of x for graphing purposes
    if extract_factor == None:
        x_fit = np.linspace(x.min(),x.max(),5000)
    elif isinstance(extract_factor, list) == True:
        x_fit = np.linspace(extract_factor[0],extract_factor[1],5000)
    y_fit = func(x_fit,*params) #plug in the 5000 x points to respective fit function to create set of S21 data for graphing

    fig = plt.figure(figurename,figsize=(15, 10))
    gs = GridSpec(6,6)
    ax1 = plt.subplot(gs[0:1,4:6]) ## original magnitude
    ax2 = plt.subplot(gs[2:3,4:6]) ## original angle
    ax3 = plt.subplot(gs[1:2,4:6]) ## normalized magnitude
    ax4 = plt.subplot(gs[3:4,4:6]) ## normalized angle
    ax = plt.subplot(gs[2:6,0:4]) ## IQ plot

    #add title
    if len(title) > 77:
        title = title[0:39] + "\n" + title[40:76] + '...'
        plt.gcf().text(0.05, 0.9, title, fontsize=30)
    if len(title) > 40:
        title = title[0:39] + "\n" + title[40:]
        plt.gcf().text(0.05, 0.9, title, fontsize=30)
    else:
        plt.gcf().text(0.05, 0.92, title, fontsize=30)

    #manual parameters
    textstr = ''
    if manual_params != None:
        if func == Cavity_inverse:
            textstr = r'Manually input parameters:' + '\n' + r'$Q_c$ = '+'%s' % float('{0:.5g}'.format(manual_params[1])) + \
            '\n' + r'$Q_i$ = '+'%s' % float('{0:.5g}'.format(manual_params[0]))+\
            '\n' + r'$f_c$ = '+'%s' % float('{0:.5g}'.format(manual_params[2]))+' GHz'\
            '\n' + r'$\phi$ = '+'%s' % float('{0:.5g}'.format(manual_params[3]))+' radians'
        elif func == Cavity_CPZM:
            textstr = r'Manually input parameters:' + '\n' + r'$Q_c$ = '+'%s' % float('{0:.5g}'.format(manual_params[0]*manual_params[1]**-1)) + \
            '\n' + r'$Q_i$ = '+'%s' % float('{0:.5g}'.format(manual_params[0]))+\
            '\n' + r'$Q_a$ = '+'%s' % float('{0:.5g}'.format(manual_params[0]*manual_params[3]**-1))+\
            '\n' + r'$f_c$ = '+'%s' % float('{0:.5g}'.format(manual_params[2]))+' GHz'
        else:
            Qc = manual_params[1]/np.exp(1j*manual_params[3])
            Qi = (manual_params[0]**-1-abs(np.real(Qc**-1)))**-1
            textstr = r'Manually input parameters:' + '\n' + 'Q = '+ '%s' % float('{0:.5g}'.format(manual_params[0])) + \
            '\n' + r'1/Re[1/$Q_c$] = ' +'%s' % float('{0:.5g}'.format(1/np.real(1/Qc))) + \
            '\n' + r'$Q_c$ = '+'%s' % float('{0:.5g}'.format(manual_params[1])) + \
            '\n' + r'$Q_i$ = '+'%s' % float('{0:.5g}'.format(Qi))+\
            '\n' + r'$f_c$ = '+'%s' % float('{0:.5g}'.format(manual_params[2]))+' GHz'+\
            '\n' + r'$\phi$ = '+'%s' % float('{0:.5g}'.format(manual_params[3]))+' radians'
        plt.gcf().text(0.1, 0.7, textstr, fontsize=15)
    else:
        plt.gcf().text(0.05, 0.85, "No manually input parameters", fontsize=15)

    #plot axies
    #ax.axvline(x=0, linewidth=1, color='grey', linestyle = '--')
    #ax.axhline(y=0, linewidth=1, color='grey', linestyle = '--')

    if isinstance(extract_factor, list) == True:
        x_fit_full = np.linspace(x.min(),x.max(),5000)
        y_fit_full = func(x_fit_full,*params)

        x_fit = np.linspace(extract_factor[0],extract_factor[1],5000)
        y_fit = func(x_fit,*params)

    if func == Cavity_inverse:
        ax1.set_ylabel('Mag[S21]')
        ax2.set_ylabel('Ang[S21]')
        ax3.set_ylabel('Mag[1/S21]')
        ax4.set_ylabel('Ang[1/S21]')
    else:
        ax1.set_ylabel('Mag[S21]')
        ax2.set_ylabel('Ang[S21]')
        ax3.set_ylabel('Mag[S21]')
        ax4.set_ylabel('Ang[S21]')

    ax1.plot(x_initial,np.log10(np.abs(y_initial))*20,'bo',label = 'raw data',color = 'black')
    ax1.plot(x,x*slope2+intercept2,'g-',label = 'normalize line',color = 'orange',linewidth = 1.5)
    ax1.set_xlim(left=x[0], right=x[-1])
    ax1.set_xlabel('frequency (GHz)')
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    ax2.plot(x_initial,np.angle(y_initial),'bo',label = 'raw data',color = 'black')
    ax2.plot(x,x*slope+intercept,'g-',label = 'normalize line',color = 'orange',linewidth = 1.5)
    ax2.set_xlim(left=x[0], right=x[-1])
    ax2.set_xlabel('frequency (GHz)')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    ax3.plot(x,np.log10(np.abs(y))*20,'bo',label = 'normalized data')
    #ax3.plot(x_fit_full,np.log10(np.abs(y_fit_full))*20,'g--',label = 'fit past 3dB from resonance',color = 'lightgreen', alpha=0.7)
    ax3.plot(x_fit,np.log10(np.abs(y_fit))*20,'g-',label = 'fit function')
    ax3.set_xlim(left=x[0], right=x[-1])
    ax3.set_xlabel('frequency (GHz)')
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    ax4.plot(x,np.angle(y),'bo',label = 'normalized data')
    #ax4.plot(x_fit_full,np.angle(y_fit_full),'g--',label = 'fit past 3dB from resonance',color = 'lightgreen', alpha=0.7)
    ax4.plot(x_fit,np.angle(y_fit),'g-',label = 'fit function')
    ax4.set_xlim(left=x[0], right=x[-1])
    ax4.set_xlabel('frequency (GHz)')
    for tick in ax4.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)


    line2 = ax.plot(np.real(y_fit),np.imag(y_fit),'g-',label = 'fit function',linewidth = 6)
    line1 = ax.plot(np.real(y),np.imag(y),'bo',label = 'normalized data',markersize = 2)
    #ax.plot(np.real(y_fit_full),np.imag(y_fit_full),'--',color = 'lightgreen',label = 'fit past 3dB from resonance',linewidth = 4.5, alpha=0.7)
    if x_c ==0 and y_c ==0 and radius == 0:
        pass
    #"""
    else:
        plt.plot(x_c,y_c,'g-',markersize = 5,color=(0, 0.8, 0.8),label = 'initial guess circle')
        circle = Circle((x_c, y_c), radius, facecolor='none',\
                    edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
        ax.add_patch(circle)
    #"""
    #plot resonance
    if func == Cavity_inverse:
        resonance = (1 + params[0]/params[1]*np.exp(1j*params[3])/(1 + 1j*2*params[0]*(params[2]-params[2])/params[2]))
    elif func == Cavity_DCM:
        resonance = 1-params[0]/params[1]*np.exp(1j*params[3])
    elif func == Cavity_DCM_REFLECTION:
        resonance = (1-2*params[0]/params[1]*np.exp(1j*params[3])/(1 + 1j*(params[2]-params[2])/params[2]*2*params[0]))
    elif func == Cavity_CPZM:
        resonance = 1/(1 + params[1] +1j*params[3])
    else:
        resonance = 1 + 1j*0
    ax.plot(np.real(resonance),np.imag(resonance),'*',color = 'red',label = 'resonance',markersize = 7)
    ax3.plot(params[2],np.log10(np.abs(resonance))*20,'*',color = 'red',label = 'resonance',markersize = 7)
    ax4.plot(params[2],np.angle(resonance),'*',color = 'red',label = 'resonance',markersize = 7)


    plt.axis('square')
    plt.ylabel('Im[S21]')
    plt.xlabel("Re[S21]")
    if func == Cavity_inverse:
         plt.ylabel('Im[$S_{21}^{-1}$]')
         plt.xlabel("Re[$S_{21}^{-1}$]")
    leg = plt.legend()
    ax1.legend(fontsize=8)

# get the individual lines inside legend and set line width
    for line in leg.get_lines():
        line.set_linewidth(10)

    try:
        if params != []:
            if func == Cavity_inverse:
                if params[0] < 0:
                    print("Qi is less than zero. Please make sure data is of correct format: decibals (log10*20 version), and radians. Otherwise, it is quite likely that the resonator being fit is not a Notch type resonator. For reflection type geometry, please use DCM REFLECTION.")
                if conf_array[0] > 10**-10 and conf_array[0] != float('inf'):
                    Qi = params[0]-params[0]%(10**int(np.log10(conf_array[0])-1))
                else:
                    Qi = params[0]
                if conf_array[1] > 10**-10 and conf_array[1] != float('inf'):
                    Qc = params[1]-params[1]%(10**int(np.log10(conf_array[1])-1))
                else:
                    Qc = params[1]
                if conf_array[2] > 10**-10 and conf_array[2] != float('inf'):
                    phi = params[3]-params[3]%(10**int(np.log10(conf_array[2])-1))
                else:
                    phi = params[3]
                if conf_array[3] > 10**-10 and conf_array[3] != float('inf'):
                    f_c = params[2]-params[2]%(10**int(np.log10(conf_array[3])-1))
                else:
                    f_c = params[2]
                textstr = r'$Q_i$ = '+'%s' % float('{0:.10g}'.format(Qi))+ r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[0]))+\
                '\n' + r'$Q_c^*$ = '+'%s' % float('{0:.10g}'.format(Qc))+ r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[1]))+\
                '\n' + r'$\phi$ = '+'%s' % float('{0:.10g}'.format(phi))+ r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[2]))+' radians'+\
                '\n' + r'$f_c$ = '+'%s' % float('{0:.10g}'.format(f_c))+ r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[3]))+' GHz'
                plt.gcf().text(0.7, 0.11, textstr, fontsize=18)
            elif func == Cavity_CPZM:
                if params[0] < 0:
                    print("Qi is less than zero. Please make sure data is of correct format: decibals (log10*20 version), and radians. Otherwise, it is quite likely that the resonator being fit is not a Notch type resonator. For reflection type geometry, please use DCM REFLECTION.")
                if conf_array[0] > 10**-10 and conf_array[0] != float('inf'):
                    Qi = params[0]-params[0]%(10**int(np.log10(conf_array[0])-1))
                else:
                    Qi = params[0]
                if conf_array[1] > 10**-10 and conf_array[1] != float('inf'):
                    Qc = (params[0]*params[1]**-1)-(params[0]*params[1]**-1)%(10**int(np.log10(conf_array[1])-1))
                else:
                    Qc = (params[0]*params[1]**-1)
                if conf_array[2] > 10**-10 and conf_array[2] != float('inf'):
                    Qa = (params[0]*params[3]**-1)-(params[0]*params[3]**-1)%(10**int(np.log10(conf_array[2])-1))
                else:
                    Qa = (params[0]*params[3]**-1)
                if conf_array[3] > 10**-10 and conf_array[3] != float('inf'):
                    f_c = params[2]-params[2]%(10**int(np.log10(conf_array[3])-1))
                else:
                    f_c = params[2]
                textstr = r'$Q_i$ = '+'%s' % float('{0:.10g}'.format(Qi))+ r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[0]))+\
                '\n' + r'$Q_c$ = '+'%s' % float('{0:.10g}'.format(Qc))+ r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[1]))+\
                '\n' + r'$Q_a$ = '+'%s' % float('{0:.10g}'.format(Qa))+ r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[2]))+\
                '\n' + r'$f_c$ = '+'%s' % float('{0:.10g}'.format(f_c))+ r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[3]))+' GHz'
                plt.gcf().text(0.7, 0.11, textstr, fontsize=18)
            else:
                Qc = params[1]/np.exp(1j*params[3])
                if Method.method == 'PHI':
                    Qi = (params[0]**-1-np.abs(Qc**-1))**-1
                else:
                    Qi = (params[0]**-1-np.real(Qc**-1))**-1

                if Qi < 0 and Method.method != 'DCM REFLECTION':
                    print("Qi is less than zero. Please make sure data is of correct format: decibals (log10*20 version), and radians. Otherwise, it is quite likely that the resonator being fit is not a Notch type resonator. Other types of resonators will not work with this code.")
                if 1/np.real(1/Qc) < 0:
                    print("Warning: 1/Real[1/Qc] is less than 0. Calculating Qi anyway")
                    if conf_array[0] > 10**-10 and conf_array[0] != float('inf'):
                        Q = params[0]-params[0]%(10**int(np.log10(conf_array[0])-1))
                    else:
                        Q = params[0]
                    if conf_array[1] > 10**-10 and conf_array[1] != float('inf'):
                        Qi = Qi-Qi%(10**int(np.log10(conf_array[1])-1))
                    if conf_array[2] > 10**-10 and conf_array[2] != float('inf'):
                        Qc = params[1]-params[1]%(10**int(np.log10(conf_array[2])-1))
                    else:
                        Qc = params[1]
                    if conf_array[3] > 10**-10 and conf_array[3] != float('inf'):
                        Qc_Re = (1/np.real(1/(params[1]/np.exp(1j*params[3]))))-(1/np.real(1/(params[1]/np.exp(1j*params[3]))))%(10**int(np.log10(conf_array[3])-1))
                    else:
                        Qc_Re = (1/np.real(1/(params[1]/np.exp(1j*params[3]))))
                    if conf_array[4] > 10**-10 and conf_array[4] != float('inf'):
                        phi = params[3]-params[3]%(10**int(np.log10(conf_array[4])-1))
                    else:
                        phi = params[3]
                    if conf_array[5] > 10**-10 and conf_array[5] != float('inf'):
                        f_c = params[2]-params[2]%(10**int(np.log10(conf_array[5])-1))
                    else:
                        f_c = params[2]
                    textstr = 'Q = '+ '%s' % float('{0:.10g}'.format(Q)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[0])) + \
                    '\n' + r'$Q_i$ = '+ '%s' % float('{0:.10g}'.format(Qi)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[1]))+\
                    '\n' + r'$Q_c$ = '+ '%s' % float('{0:.10g}'.format(Qc)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[2])) + \
                    '\n' + r'$\phi$ = '+'%s' % float('{0:.10g}'.format(phi)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[4]))+' radians'+\
                    '\n' + r'$f_c$ = '+'%s' % float('{0:.10g}'.format(f_c)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[5]))+' GHz'
                    plt.gcf().text(0.7, 0.09, textstr, fontsize=18)
                    Qc_str = r'1/Re[1/$Q_c$] = ' +'%s' % float('{0:.10g}'.format(Qc_Re)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[3]))
                    plt.gcf().text(0.7, 0.245, Qc_str, fontsize=18, color = 'red')

                else:
                    if conf_array[0] > 10**-10 and conf_array[0] != float('inf'):
                        Q = params[0]-params[0]%(10**int(np.log10(conf_array[0])-1))
                    else:
                        Q = params[0]
                    if conf_array[1] > 10**-10 and conf_array[1] != float('inf'):
                        Qi = Qi-Qi%(10**int(np.log10(conf_array[1])-1))
                    if conf_array[2] > 10**-10 and conf_array[2] != float('inf'):
                        Qc = params[1]-params[1]%(10**int(np.log10(conf_array[2])-1))
                    else:
                        Qc = params[1]
                    if conf_array[3] > 10**-10 and conf_array[3] != float('inf'):
                        Qc_Re = (1/np.real(1/(params[1]/np.exp(1j*params[3]))))-(1/np.real(1/(params[1]/np.exp(1j*params[3]))))%(10**int(np.log10(conf_array[3])-1))
                    else:
                        Qc_Re = (1/np.real(1/(params[1]/np.exp(1j*params[3]))))
                    if conf_array[4] > 10**-10 and conf_array[4] != float('inf'):
                        phi = params[3]-params[3]%(10**int(np.log10(conf_array[4])-1))
                    else:
                        phi = params[3]
                    if conf_array[5] > 10**-10 and conf_array[5] != float('inf'):
                        f_c = params[2]-params[2]%(10**int(np.log10(conf_array[5])-1))
                    else:
                        f_c = params[2]
                    textstr = 'Q = '+ '%s' % float('{0:.10g}'.format(Q)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[0])) + \
                    '\n' + r'$Q_i$ = '+ '%s' % float('{0:.10g}'.format(Qi)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[1]))+\
                    '\n' + r'$Q_c$ = '+ '%s' % float('{0:.10g}'.format(Qc)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[2])) + \
                    '\n' + r'1/Re[1/$Q_c$] = ' +'%s' % float('{0:.10g}'.format(Qc_Re)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[3])) + \
                    '\n' + r'$\phi$ = '+'%s' % float('{0:.10g}'.format(phi)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[4]))+' radians'+\
                    '\n' + r'$f_c$ = '+'%s' % float('{0:.10g}'.format(f_c)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[5]))+' GHz'
                    plt.gcf().text(0.7, 0.09, textstr, fontsize=18)

            #write to output csv file
            title_without_period = ''
            for i in title: #remove period from title
                if i != '.':
                    title_without_period = title_without_period + i
            file = open(output_path + title_without_period + "_output.csv","w")
            if func == Cavity_inverse:
                textstr = r'Qi = '+'%s' % float('{0:.10g}'.format(Qi))+ r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[0]))+\
                '\n' + r'Qc* = '+'%s' % float('{0:.10g}'.format(Qc))+ r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[1]))+\
                '\n' + r'phi = '+'%s' % float('{0:.10g}'.format(phi))+ r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[2]))+' radians'+\
                '\n' + r'fc = '+'%s' % float('{0:.10g}'.format(f_c))+ r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[3]))+' GHz'
            elif func == Cavity_CPZM:
                textstr = r'Q_i = '+'{0:01f}'.format(params[0]) + \
                '\n' + r'Q_c = '+'{0:01f}'.format(params[0]*params[1]**-1)+\
                '\n' + r'Q_a = '+'{0:01f}'.format(params[0]*params[3]**-1)+\
                '\n' + r'f_c = '+'{0:01f}'.format(params[2])+' GHz'

                textstr = r'Qi = '+'%s' % float('{0:.10g}'.format(Qi))+ r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[0]))+\
                '\n' + r'Qc = '+'%s' % float('{0:.10g}'.format(Qc))+ r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[1]))+\
                '\n' + r'Qa = '+'%s' % float('{0:.10g}'.format(Qa))+ r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[2]))+\
                '\n' + r'fc = '+'%s' % float('{0:.10g}'.format(f_c))+ r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[3]))+' GHz'
            else:
                textstr = 'Q = '+ '%s' % float('{0:.10g}'.format(Q)) + r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[0])) + \
                '\n' + r'Qi = '+ '%s' % float('{0:.10g}'.format(Qi)) + r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[1]))+\
                '\n' + r'Qc = '+ '%s' % float('{0:.10g}'.format(Qc)) + r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[2])) + \
                '\n' + r'1/Re[1/Qc] = ' +'%s' % float('{0:.10g}'.format(Qc_Re)) + r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[3])) + \
                '\n' + r'phi = '+'%s' % float('{0:.10g}'.format(phi)) + r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[4]))+' radians'+\
                '\n' + r'fc = '+'%s' % float('{0:.10g}'.format(f_c)) + r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[5]))+' GHz'
        file.write(textstr)
    except:
        print(">Error when trying to write parameters on plot")
        quit()

    plt.tight_layout()
    return fig

#############################################################################
def fit_raw_compare(x,y,params,method):
    if method == 'DCM':
        func = ff.Cavity_DCM
    if method == 'INV':
        func = ff.Cavity_inverse
    yfit = func(x,*params)
    ym = np.abs(y-yfit)/np.abs(y)
    return ym
############################################################################
    ## Least square fit model for one cavity, dip-like S21
def min_one_Cavity_dip(parameter, x, data=None):
    #fit function call for DCM fitting method
    Q = parameter['Q']
    Qc = parameter['Qc']
    w1 = parameter['w1']
    phi = parameter['phi']

    model = ff.Cavity_DCM(x, Q, Qc, w1,phi)
    real_model = model.real
    imag_model = model.imag
    real_data = data.real
    imag_data = data.imag

    resid_re = real_model - real_data
    resid_im = imag_model - imag_data
    return np.concatenate((resid_re,resid_im))

########################################################

    ## Least square fit model for one cavity, dip-like S21
def min_one_Cavity_DCM_REFLECTION(parameter, x, data=None):
    #fit function call for DCM fitting method
    Q = parameter['Q']
    Qc = parameter['Qc']
    w1 = parameter['w1']
    phi = parameter['phi']

    model = ff.Cavity_DCM_REFLECTION(x, Q, Qc, w1,phi)
    real_model = model.real
    imag_model = model.imag
    real_data = data.real
    imag_data = data.imag

    resid_re = real_model - real_data
    resid_im = imag_model - imag_data
    return np.concatenate((resid_re,resid_im))

########################################################
    ## Least square fit model for one cavity, dip-like "Inverse" S21
def min_one_Cavity_inverse(parameter, x, data=None):
    #fit function call for INV fitting method
    Qc = parameter['Qc']
    Qi = parameter['Qi']
    w1 = parameter['w1']
    phi = parameter['phi']

    model = ff.Cavity_inverse(x, Qi, Qc, w1,phi)
    real_model = model.real
    imag_model = model.imag
    real_data = data.real
    imag_data = data.imag

    resid_re = real_model - real_data
    resid_im = imag_model - imag_data
    return np.concatenate((resid_re,resid_im))


#####################################################################
def min_one_Cavity_CPZM(parameter, x, data=None):
    #fit function call for CPZM fitting method
    Qc = parameter['Qc']
    Qi = parameter['Qi']
    w1 = parameter['w1']
    Qa = parameter['Qa']

    model = ff.Cavity_CPZM(x, Qi, Qc, w1,Qa)
    real_model = model.real
    imag_model = model.imag
    real_data = data.real
    imag_data = data.imag

    resid_re = real_model - real_data
    resid_im = imag_model - imag_data
    return np.concatenate((resid_re,resid_im))


#####################################################################
def monte_carlo_fit(xdata=None, ydata=None, parameter=None, Method=None):
    print("Start MCFit")
    try:
        ydata_1stfit = Method.func(xdata, *parameter) #set of S21 data based on initial guess parameters

        ## weight condition
        if Method.MC_weight == 'yes':
            weight_array = 1/abs(ydata) #new array of inversed magnitude ydata
        else:
            weight_array = np.full(len(xdata),1) #new array of len(xdata) all slots filled with 1

        weighted_ydata = np.multiply(weight_array,ydata) #array filled with 1s if MC_weight='yes' and exact same array as ydata otherwise
        weighted_ydata_1stfit = np.multiply(weight_array,ydata_1stfit) #array with values (ydata^(-1))*ydata_1stfit if MC_weight='yes' and exact same array as ydata_1stfit otherwise
        error = np.linalg.norm(weighted_ydata - weighted_ydata_1stfit)/len(xdata) # first error #finds magnitude of (weighted_ydata-weighted_ydata_1stfit) and divides by length (average magnitude)
        error_0 = error
        print("Initial Error: ", error_0)
    except:
        raise Exception(">Failed to initialize MonteCarloFit(), please check parameters")
    ## Fix condition and Monte Carlo Method with random number Generator

    counts = 0
    try:
        while counts < Method.MC_rounds: #MC_rounds 100,000 by default
            counts = counts + 1
            #generate an array of 4 random numbers between -0.5 and 0.5 in the format [r,r,r,r] where r is each of the random numbers times the step constant
            random = Method.MC_step_const*(np.random.random_sample(len(parameter))-0.5)
           ## Fix parameter on demand
            if 'Q' in Method.MC_fix:
                random[0] = 0
            if 'Qi' in Method.MC_fix:
                random[0] = 0
            if 'Qc' in Method.MC_fix:
                random[1] = 0
            if 'w1' in Method.MC_fix: #in MC_fix
                random[2] = 0
            if 'phi' in Method.MC_fix:
                random[3] = 0
            if 'Qa' in Method.MC_fix:
                random[3] = 0
            ## Generate new parameter to test
            random[3] = random[3]*0.1
            random = np.exp(random) #not really that even of a distribution
            new_parameter = np.multiply(parameter,random)
            #if Method.method == 'CPZM':
            #    new_parameter[3] = new_parameter[3]/np.exp(0.1)
            #else:
            new_parameter[3] = np.mod(new_parameter[3], 2*np.pi) # phi from 0 to 2*pi

            ydata_MC = Method.func(xdata, *new_parameter) #new set of data with new parameters
            #check new error with new set of parameters
            weighted_ydata_MC = np.multiply(weight_array, ydata_MC)
            new_error = np.linalg.norm(weighted_ydata_MC - weighted_ydata)/len(xdata)
            if new_error < error:
                parameter = new_parameter
                error = new_error
                print("New Error: ", new_error, counts)
    except:
        raise Exception(">Error in while loop of MonteCarloFit")

   ## If finally gets better fit then plot ##
    if error < error_0:
        stop_MC = False
        print('Monte Carlo fit got better fitting parameters')
        if Method.manual_init != None:
            print('>User input parameters getting stuck in local minimum, please input more accurate parameters')
    else:
        stop_MC = True
    return parameter, stop_MC, error

## Fit data to least squares fit for respective fit type
def min_fit(params,xdata,ydata,Method):
    try:
        if Method.method == 'DCM' or Method.method == 'PHI':
            minner = Minimizer(min_one_Cavity_dip, params, fcn_args=(xdata, ydata))
        elif Method.method == 'DCM REFLECTION':
            minner = Minimizer(min_one_Cavity_DCM_REFLECTION, params, fcn_args=(xdata, ydata))
        elif Method.method == 'INV':
            minner = Minimizer(min_one_Cavity_inverse, params, fcn_args=(xdata, ydata))
        elif Method.method == 'CPZM':
            minner = Minimizer(min_one_Cavity_CPZM, params, fcn_args=(xdata, ydata))

        result = minner.minimize(method = 'least_squares')

        fit_params = result.params
        parameter = fit_params.valuesdict()
        fit_params = [value for _,value in parameter.items()] #extracts the actual value for each parameter and puts it in the fit_params list

        if Method.method == 'DCM' or Method.method == 'PHI' or Method.method == 'DCM REFLECTION':
            ci = lmfit.conf_interval(minner, result, p_names=['Q','Qc','phi','w1'], sigmas=[2])
            #confidence interval for Q
            Q_conf = max(np.abs(ci['Q'][1][1]-ci['Q'][0][1]),np.abs(ci['Q'][1][1]-ci['Q'][2][1]))
            #confidence interval for Qi
            if Method.method == 'PHI':
                Qi = ((ci['Q'][1][1])**-1 - np.abs(ci['Qc'][1][1]**-1 * np.exp(1j*fit_params[3])))**-1
                Qi_neg = Qi-((ci['Q'][0][1])**-1 - np.abs(ci['Qc'][2][1]**-1 * np.exp(1j*fit_params[3])))**-1
                Qi_pos = Qi-((ci['Q'][2][1])**-1 - np.abs(ci['Qc'][0][1]**-1 * np.exp(1j*fit_params[3])))**-1
            else:
                Qi = ((ci['Q'][1][1])**-1 - np.real(ci['Qc'][1][1]**-1 * np.exp(1j*fit_params[3])))**-1
                Qi_neg = Qi-((ci['Q'][0][1])**-1 - np.real(ci['Qc'][2][1]**-1 * np.exp(1j*fit_params[3])))**-1
                Qi_pos = Qi-((ci['Q'][2][1])**-1 - np.real(ci['Qc'][0][1]**-1 * np.exp(1j*fit_params[3])))**-1
            Qi_conf = max(np.abs(Qi_neg),np.abs(Qi_pos))
            #confidence interval for Qc
            Qc_conf = max(np.abs(ci['Qc'][1][1]-ci['Qc'][0][1]),np.abs(ci['Qc'][1][1]-ci['Qc'][2][1]))
            #confidence interval for 1/Re[1/Qc]
            Qc_Re = 1/np.real(np.exp(1j*fit_params[3])/ci['Qc'][1][1])
            Qc_Re_neg = 1/np.real(np.exp(1j*fit_params[3])/ci['Qc'][0][1])
            Qc_Re_pos = 1/np.real(np.exp(1j*fit_params[3])/ci['Qc'][2][1])
            Qc_Re_conf = max(np.abs(Qc_Re-Qc_Re_neg),np.abs(Qc_Re-Qc_Re_pos))
            #confidence interval for phi
            phi_neg = ci['phi'][0][1]
            phi_pos = ci['phi'][2][1]
            phi_conf = max(np.abs(ci['phi'][1][1]-ci['phi'][0][1]),np.abs(ci['phi'][1][1]-ci['phi'][2][1]))
            #confidence interval for resonance frequency
            w1_neg = ci['w1'][0][1]
            w1_pos = ci['w1'][2][1]
            w1_conf = max(np.abs(ci['w1'][1][1]-ci['w1'][0][1]),np.abs(ci['w1'][1][1]-ci['w1'][2][1]))
            #Array of confidence intervals
            conf_array = [Q_conf,Qi_conf,Qc_conf,Qc_Re_conf,phi_conf,w1_conf]
        elif Method.method == 'INV':
            ci = lmfit.conf_interval(minner, result, p_names=['Qi','Qc','phi','w1'], sigmas=[2])
            #confidence interval for Qi
            Qi_conf = max(np.abs(ci['Qi'][1][1]-ci['Qi'][0][1]),np.abs(ci['Qi'][1][1]-ci['Qi'][2][1]))
            #confidence interval for Qc
            Qc_conf = max(np.abs(ci['Qc'][1][1]-ci['Qc'][0][1]),np.abs(ci['Qc'][1][1]-ci['Qc'][2][1]))
            #confidence interval for phi
            phi_conf = max(np.abs(ci['phi'][1][1]-ci['phi'][0][1]),np.abs(ci['phi'][1][1]-ci['phi'][2][1]))
            #confidence interval for resonance frequency
            w1_conf = max(np.abs(ci['w1'][1][1]-ci['w1'][0][1]),np.abs(ci['w1'][1][1]-ci['w1'][2][1]))
            #Array of confidence intervals
            conf_array = [Qi_conf,Qc_conf,phi_conf,w1_conf]
        else:
            ci = lmfit.conf_interval(minner, result, p_names=['Qi','Qc','Qa','w1'], sigmas=[2])
            #confidence interval for Qi
            Qi_conf = max(np.abs(ci['Qi'][1][1]-ci['Qi'][0][1]),np.abs(ci['Qi'][1][1]-ci['Qi'][2][1]))
            #confidence interval for Qc
            Qc = ci['Qi'][1][1]/ci['Qc'][1][1]
            Qc_neg = ci['Qi'][0][1]/ci['Qc'][0][1]
            Qc_pos = ci['Qi'][2][1]/ci['Qc'][2][1]
            Qc_conf = max(np.abs(Qc-Qc_neg),np.abs(Qc-Qc_neg))
            #confidence interval for Qa
            Qa = ci['Qi'][1][1]/ci['Qa'][1][1]
            Qa_neg = ci['Qi'][2][1]/ci['Qa'][2][1]
            Qa_pos = ci['Qi'][0][1]/ci['Qa'][0][1]
            Qa_conf = max(np.abs(Qa-Qa_neg),np.abs(Qa-Qa_neg))
            #confidence interval for resonance frequency
            w1_conf = max(np.abs(ci['w1'][1][1]-ci['w1'][0][1]),np.abs(ci['w1'][1][1]-ci['w1'][2][1]))
            #Array of confidence intervals
            conf_array = [Qi_conf,Qc_conf,Qa_conf,w1_conf]

        return fit_params, conf_array
    except:
        print(">Failed to minimize function for least squares fit")
        quit()


@attr.dataclass(frozen=True)
class VNASweep:
    """A container to hold data from a vna frequency sweep."""
    freqs: np.ndarray
    amps: np.ndarray
    phases: np.ndarray
    linear_amps: np.ndarray=None

    @classmethod
    def from_csv(cls, csv):
        """Load data from csv file."""
        data = np.loadtxt(csv, delimiter=',')
        freqs = data.T[0]
        amps = data.T[1]
        phases = data.T[2]
        linear_amps = 10**(amps/20)
        return cls(freqs=freqs,amps=amps,phases=phases,linear_amps=linear_amps)

    @classmethod
    def from_columns(cls, freqs, amps, phases):
        """Load data from columns provided individually."""
        linear_amps = 10 ** (amps / 20)
        return cls(freqs=freqs, amps=amps, phases=phases, linear_amps=linear_amps)


@attr.dataclass(frozen=True)
class ComplexData:
    """Container for normalized data"""
    freqs: np.ndarray
    complex_s21: np.ndarray

def normalize_data(data: VNASweep,
                   background: VNASweep = None)->ComplexData:
    """Normalize and exponentiate data.

    Also performs background subtraction if applicable.

    Args:
        freq: Frequencies of scan. Units of GHz.
        amplitude: Amplitude data from vna, units of dB
        phase: Phase data from vna, in radians.
        background (optional): 3 column array, frequencies, amplitudes and
          phase.

    Returns:
        Normalized data complex S21.
    """

    xdata = data.freqs
    linear_amps = data.linear_amps
    phases = data.phases
    complex_data = np.multiply(linear_amps, np.exp(1j * phases))

    if background:
        amps_background = background.linear_amps
        phases_background = background.phases

        amps_subtracted = np.divide(linear_amps, amps_background)
        phases_subtracted = np.subtract(phases, phases_background)
        complex_data = np.multiply(amps_subtracted,
                                   np.exp(1j * phases_subtracted))

    complex_data = ComplexData(freqs=xdata, complex_s21=complex_data)
    return complex_data

def preprocess(cplx_data: ComplexData, normalize: int)->ComplexData:
    """Preprocess data by removing cable delay and normalizing S21.

    Args:
        cplx_data: Complex data, frequency vs. A*exp(j*phi)
        normalize: Number of at beginning and end of trace for normalization.

    Returns:
        Data with cable delay removed and normalized amplitudes.
    """

    x_initial = cplx_data.freqs
    y_initial = cplx_data.complex_s21

    if normalize * 2 > len(y_initial):
        raise ValueError(
            'Not enough points to normalize, please lower value of normalize variable or take more points near resonance')

    # normalize phase of S21 using linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.append(x_initial[0:10], x_initial[-10:]),
        np.append(np.angle(y_initial[0:normalize]),
                  np.angle(y_initial[-normalize:])))
    angle = np.subtract(np.angle(y_initial), slope * x_initial)  # remove cable delay
    angle = np.subtract(angle, intercept)  # rotate off resonant point to (1,0i) in complex plane

    # normalize magnitude of S21 using linear fit
    y_db = np.log10(np.abs(y_initial)) * 20
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(
        np.append(x_initial[0:normalize], x_initial[-normalize:]),
        np.append(y_db[0:normalize], y_db[-normalize:]))
    magnitude = np.subtract(y_db, slope2 * x_initial + intercept2)
    magnitude = 10 ** (magnitude / 20)

    y_raw = np.multiply(magnitude, np.exp(1j * angle))
    preped_data = ComplexData(x_initial, y_raw)
    return preped_data

def extract_near_res(x_raw: np.ndarray,
                     y_raw: np.ndarray,
                     f_res: float,
                     kappa: float,
                     extract_factor: int=1)->ComplexData:
    """Extracts portions of spectrum of kappa within resonance.

    Args:
        x_raw: X-values of spectrum to extract from.
        y_raw: Y-values of spectrum to extract from.
        f_res: Resonant frequency about which to extract data.
        kappa: Width about f_res to extract.
        extract_factor: Multiplier for kappa.

    Returns:
        Extracted spectrum kappa about f_res.
    """
    xstart = f_res - extract_factor/2*kappa #starting resonance to add to fit
    xend = f_res + extract_factor/2*kappa #final resonance to add to fit
    x_temp = []
    y_temp = []
    # xdata is new set of data to be fit, within extract_factor times the bandwidth, ydata is S21 data to match indices with xdata
    for i, freq in enumerate(x_raw):
        if (freq > xstart and freq< xend):
            x_temp.append(freq)
            y_temp.append(y_raw[i])

    if len(y_temp) < 5:
        print("Less than 5 Data points to fit data, not enough points near resonance, attempting to fit anyway")
    if len(x_temp) < 1:
        raise Exception(">Failed to extract data from designated bandwidth")

    return np.asarray(x_temp), np.asarray(y_temp)

def Fit_Resonator(filename,data_array,Method,normalize,background = None):
    data = VNASweep.from_columns(freqs=data_array.T[0],
                                 amps=data_array.T[1],
                                 phases=data_array.T[2])

    normed_data = normalize_data(data, background=background)

    print("Loaded the data!")

    prepped_data = preprocess(normed_data, normalize=normalize)
    resonator = Resonator(prepped_data.freqs,
                          prepped_data.complex_s21,
                          name=filename)


    ## Init function variables
    manual_init = Method.manual_init
    find_circle = Method.find_circle
    vary = Method.vary
    filename = resonator.name
    xdata = resonator.freq
    ydata = resonator.S21
    y1data = np.real(ydata)
    y2data = np.imag(ydata)
    x_raw = resonator.freq
    y_raw = resonator.S21

    ##### Step one. Find initial guess if not specified and extract part of data close to resonance  #####

    if len(x_raw) < 20:
        raise ValueError(">Not enough data points to run code. Please have at least 20 data points.")

    if manual_init: #when user manually initializes a guess initialize the following variables
        try:
            init = manual_init
            if Method.method.name in ['DCM', "DCM_REFLECTION" ,'PHI']:
                # If method is DCM or PHI, set parameter Q to 1/(1/Qi + 1/Qc) aka. convert from Qi
                init.Q = 1/(1/init.Qi + 1/init.Qc)
            elif Method.method.name == 'CPZM':
                # @mullinski please comment on this.
                init.Qa = init.Qi/init.f_res
                init.Qc = init.Qi/init.Qc
                init.f_res = init.Qi/init.phi

            freq = init.f_res
            kappa = init.f_res/(init.Qi) #bandwidth for frequency values
            x_c, y_c, r = 0,0,0 #set initial guess circle variables to zero so circle does not appear in plots
            print("Manual initial guess")
        except:
            raise Exception(">Problem loading manually initialized parameters, please make sure parameters are all numbers")

    else: # TODO(mutus) Take care of case with no initial guess.
        #generate initial guess parameters from data when user does not manually initialze guess
        init, x_c, y_c, r = Find_initial_guess(xdata, y1data, y2data, Method) #gets initial guess for parameters
        freq = init[2] #resonance frequency
        kappa = init[2]/init[0] #f_0/Qi is kappa
        if Method.method == 'CPZM':
            kappa = init[2]*init[1]/init[0]

    ## Extract data near resonant frequency to fit
    xdata, ydata = extract_near_res(x_raw, y_raw, freq, kappa)

    if Method.method == 'INV':
        ydata = ydata**-1 ## Inverse S21

    #####==== Step Two. Fit Both Re and Im data  ====####
        # create a set of Parameters
        ## Monte Carlo Loop for inverse S21

    #define parameters from initial guess for John Martinis and MonteCarloFit
    try:
        params = lmfit.Parameters() #initialize parameter class, min is lower bound, max is upper bound, vary = boolean to determine if parameter varies during fit
        if Method.method.name in ['DCM', 'DCM_REFLECTION', 'PHI']:
            params.add('Q', value=init.Q, vary = vary[0], min = init.Q*0.5, max = init.Q*1.5)
        elif Method.method.name in ['INV', 'CPZM']:
            params.add('Qi', value=init.Qi, vary = vary[0], min = init.Qi*0.8, max = init.Qi*1.2)
        params.add('Qc', value=init.Qc, vary = vary[1], min = init.Qc*0.8, max = init.Qc*1.2)
        params.add('w1', value=init.f_res, vary = vary[2], min = init.f_res*0.9, max = init.f_res*1.1)
        if Method.method == 'CPZM':
            params.add('Qa', value=init.Qa, vary = vary[3] , min=init.Qa*0.9, max=init.Qa*1.1)
        else:
            params.add('phi', value=init.phi, vary = vary[3] , min = init.phi*0.9, max=init.phi*1.1)
        print("Parameters: ", params)
    except:
        raise ValueError(">Failed to define parameters, please make sure parameters are of correct format")

    #setup for while loop
    MC_counts = 0
    error = [10]
    stop_MC = False
    continue_condition = (MC_counts < Method.MC_iteration) and (stop_MC == False) #MC_iteration equals 5 by default
    output_params = []
    print("About to do some runs on data len: ", len(xdata), len(ydata))

    while continue_condition: #will run exactly 5 times unless error encountered
        ## Fit data to least squares fit for respective fit type
        try:
            if Method.method.name in ['DCM' ,'PHI']:
                print("I am the DCM or PHI: ", Method.method.name)
                minner = Minimizer(min_one_Cavity_dip, params, fcn_args=(xdata, ydata))
            elif Method.method.name == 'DCM REFLECTION':
                minner = Minimizer(min_one_Cavity_DCM_REFLECTION, params, fcn_args=(xdata, ydata))
            elif Method.method.name == 'INV':
                minner = Minimizer(min_one_Cavity_inverse, params, fcn_args=(xdata, ydata))
            elif Method.method.name == 'CPZM':
                minner = Minimizer(min_one_Cavity_CPZM, params, fcn_args=(xdata, ydata))

            result = minner.minimize(method = 'least_squares')
            fit_params = result.params
            parameter = fit_params.valuesdict()

            fit_params = [value for _,value in parameter.items()] #extracts the actual value for each parameter and puts it in the fit_params list
        except:
            raise ValueError(">Failed to minimize function for least squares fit")

    #####==== Try Monte Carlo Fit Inverse S21 ====####

        MC_param, stop_MC, error_MC = \
        monte_carlo_fit(xdata, ydata, fit_params, Method) #run a Monte Carlo fit on just minimized data to test if parameters trapped in local minimum
        error.append(error_MC)
        if error[MC_counts] < error_MC:
            stop_MC = True

        output_params.append(MC_param)
        MC_counts = MC_counts+1

        continue_condition = (MC_counts < Method.MC_iteration) and (stop_MC == False)

        if continue_condition == False:
            output_params = output_params[MC_counts-1]

    error = min(error)

    #Check that bandwidth is not equal to zero
    if len(xdata) == 0:
        if manual_init != None:
            print(">Length of extracted data equals zero thus bandwidth is incorrect, most likely due to initial parameters being too far off")
            print(">Please enter a new set of manual initial guess data or run an auto guess")
        else:
            raise ValueError(">Length of extracted data equals zero thus bandwidth is incorrect, please manually input a guess for parameters")

    #set the range to plot for 1 3dB bandwidth
    if Method.method == 'CPZM':
        Q = 1/(1/output_params[0] + output_params[1]/output_params[0])
        kappa = output_params[2]/Q
    else:
        kappa = output_params[2]/output_params[0]
    xstart = output_params[2] - kappa/2 #starting resonance to add to fit
    xend = output_params[2] + kappa/2
    extract_factor = [xstart,xend]

    #plot fit
    # if Method.method == 'DCM':
    #     try:
    #         title = 'DCM fit for ' + filename
    #         figurename =" DCM with Monte Carlo Fit and Raw data\nPower: " + filename
    #         fig = PlotFit(x_raw,y_raw,x_initial,y_initial,slope,intercept,slope2,intercept2,output_params,Method,Cavity_DCM,error,figurename,x_c,y_c,r,output_path,extract_factor,title = title, manual_params = Method.manual_init)
    #     except:
    #         print(">Failed to plot DCM fit for data")
    #         quit()
    # if Method.method == 'PHI':
    #     try:
    #         title = 'PHI fit for ' + filename
    #         figurename ="PHI with Monte Carlo Fit and Raw data\nPower: " + filename
    #         fig = PlotFit(x_raw,y_raw,x_initial,y_initial,slope,intercept,slope2,intercept2,output_params,Method,Cavity_DCM,error,figurename,x_c,y_c,r,output_path,extract_factor,title = title, manual_params = Method.manual_init)
    #     except:
    #         print(">Failed to plot PHI fit for data")
    #         quit()
    # if Method.method == 'DCM REFLECTION':
    #     try:
    #         title = 'DCM REFLECTION fit for ' + filename
    #         figurename =" DCM REFLECTION with Monte Carlo Fit and Raw data\nPower: " + filename
    #         fig = PlotFit(x_raw,y_raw,x_initial,y_initial,slope,intercept,slope2,intercept2,output_params,Method,Cavity_DCM_REFLECTION,error,figurename,x_c,y_c,r,output_path,extract_factor,title = title, manual_params = Method.manual_init)
    #     except:
    #         print(">Failed to plot DCM fit for data")
    #         quit()
    # elif Method.method == 'INV':
    #     try:
    #         title = 'INV fit for ' + filename
    #         figurename = " Inverse with MC Fit and Raw data\nPower: " + filename
    #         fig = PlotFit(x_raw,1/y_raw,x_initial,y_initial,slope,intercept,slope2,intercept2,output_params,Method,Cavity_inverse,error,figurename,x_c,y_c,r,output_path,extract_factor,title = title, manual_params = Method.manual_init)
    #     except:
    #         print(">Failed to plot INV fit for data")
    #         quit()
    # elif Method.method == 'CPZM':
    #     try:
    #         title = 'CPZM fit for ' + filename
    #         figurename = " CPZM with MC Fit and Raw data\nPower: " + filename
    #         fig = PlotFit(x_raw,y_raw,x_initial,y_initial,slope,intercept,slope2,intercept2,output_params,Method,Cavity_CPZM,error,figurename,x_c,y_c,r,output_path,extract_factor,title = title, manual_params = Method.manual_init)
    #     except:
    #         print(">Failed to plot CPZM fit for data")
    #         quit()
    # fig.savefig(output_path+filename+'_'+Method.method+'_fit.png')
    # return output_params,fig,error,init
    return output_params,error,init

def plot(x,y,name,output_path,x_c=None,y_c=None,r=None,p_x=None,p_y=None):
    #plot any given set of x and y data
    fig = plt.figure('raw_data',figsize=(10, 10))
    gs = GridSpec(2,2)
    ax = plt.subplot(gs[0:2,0:2]) ## plot
    #plot axies
    #ax.axvline(x=0, linewidth=1, color='grey', linestyle = '--')
    #ax.axhline(y=0, linewidth=1, color='grey', linestyle = '--')
    ax.plot(x,y,'bo',label = 'raw data',markersize = 3)
    #plot guide circle if it applies
    if x_c!=None and y_c!=None and r!=None:
        circle = Circle((x_c, y_c), r, facecolor='none',\
                    edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
        ax.add_patch(circle)
    #plot a red point to represent something if it applies (resonance or off resonance for example)
    if p_x!=None and p_y!=None:
        ax.plot(p_x,p_y,'*',color = 'red',markersize = 5)
    fig.savefig(output_path+name+'.png')

#########################################################################

def plot2(x,y,x2,y2,name,output_path):
    #plot any given set of x and y data
    fig = plt.figure('raw_data',figsize=(10, 10))
    gs = GridSpec(2,2)
    ax = plt.subplot(gs[0:2,0:2]) ## plot
    ax.plot(x,y,'bo',label = 'raw data',markersize = 3)
    ax.plot(x2,y2,'bo',label = 'raw data',markersize = 3, color = 'red')
    fig.savefig(output_path+name+'.png')
